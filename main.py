import os
import io
from typing import Optional, Dict
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ------------------------------
# Predictor class (aligned with notebook)
# ------------------------------
class BPFPredictor:
    """Loads BPF models and artifacts and makes predictions."""

    def __init__(self, s21_model=None, s11_model=None, scaler=None, imputer=None, features=None):
        self.s21_model = s21_model
        self.s11_model = s11_model
        self.scaler = scaler
        self.imputer = imputer
        self.features = features

    @classmethod
    def from_directory(cls, model_dir: str):
        """Load artifacts from a directory path with standard filenames."""
        try:
            s21 = joblib.load(os.path.join(model_dir, 'bpf_s21_model.pkl'))
            s11 = joblib.load(os.path.join(model_dir, 'bpf_s11_model.pkl'))
            scaler = joblib.load(os.path.join(model_dir, 'bpf_scaler.pkl'))
            imputer = joblib.load(os.path.join(model_dir, 'bpf_imputer.pkl'))
            features = joblib.load(os.path.join(model_dir, 'bpf_features.joblib'))
            return cls(s21, s11, scaler, imputer, features)
        except Exception as e:
            raise RuntimeError(f"Failed to load artifacts from {model_dir}: {e}")

    @classmethod
    def from_uploaded_files(cls, s21_file, s11_file, scaler_file, imputer_file, features_file):
        """Load artifacts from uploaded files (Streamlit UploadedFile)."""
        try:
            s21 = joblib.load(io.BytesIO(s21_file.read()))
            s11 = joblib.load(io.BytesIO(s11_file.read()))
            scaler = joblib.load(io.BytesIO(scaler_file.read()))
            imputer = joblib.load(io.BytesIO(imputer_file.read()))
            features = joblib.load(io.BytesIO(features_file.read()))
            return cls(s21, s11, scaler, imputer, features)
        except Exception as e:
            raise RuntimeError(f"Failed to load uploaded artifacts: {e}")

    def is_ready(self) -> bool:
        return all([self.s21_model is not None, self.s11_model is not None, self.scaler is not None, self.imputer is not None, self.features is not None])

    def _create_feature_dataframe(self, data_dict: Dict[str, Optional[float]]) -> pd.DataFrame:
        """Creates a dataframe with features in the correct order."""
        template_dict = {feature: np.nan for feature in self.features}
        template_dict.update(data_dict)
        return pd.DataFrame([template_dict], columns=self.features)

    def predict(self, *, frequency_hz: float, f1_mhz: float, f2_mhz: float, z0_ohm: float = 50.0,
                f3_mhz: Optional[float] = None, rej_f3_db: Optional[float] = None,
                f4_mhz: Optional[float] = None, rej_f4_db: Optional[float] = None) -> dict:
        if not self.is_ready():
            return {"Error": "Predictor not initialized."}

        # 1) Replicate feature engineering used in training
        input_data: Dict[str, Optional[float]] = {
            'Passband F1 (MHz)': f1_mhz,
            'Passband F2 (MHz)': f2_mhz,
            'frequency_hz': frequency_hz,
            'z0_ohm': z0_ohm,
            'Stopband F3 (MHz)': f3_mhz,
            'Rejection @ F3 (dB)': rej_f3_db,
            'Stopband F4 (MHz)': f4_mhz,
            'Rejection @ F4 (dB)': rej_f4_db,
        }

        f_center = (f1_mhz + f2_mhz) / 2 if (f1_mhz is not None and f2_mhz is not None) else np.nan
        bw = (f2_mhz - f1_mhz) if (f1_mhz is not None and f2_mhz is not None) else np.nan
        input_data['f_center'] = f_center
        input_data['BW'] = bw
        input_data['frac_BW'] = (bw / f_center) if f_center not in (0, None) and not pd.isna(f_center) else np.nan
        input_data['Q_approx'] = (f_center / bw) if bw not in (0, None) and not pd.isna(bw) else np.nan
        input_data['log_f_center'] = np.log10(f_center * 1e6) if f_center is not None and f_center > 0 else np.nan

        input_data['stop_offset1'] = (f1_mhz - f3_mhz) if (f3_mhz is not None and f1_mhz is not None) else np.nan
        input_data['stop_offset2'] = (f4_mhz - f2_mhz) if (f4_mhz is not None and f2_mhz is not None) else np.nan
        input_data['F3_norm'] = (f3_mhz / f_center) if (f3_mhz is not None and f_center not in (0, None) and not pd.isna(f_center)) else np.nan

        freq_diff1 = (f1_mhz - f3_mhz) if (f3_mhz is not None and f1_mhz is not None) else np.nan
        input_data['slope1'] = (rej_f3_db / freq_diff1) if (rej_f3_db is not None and not pd.isna(freq_diff1) and freq_diff1 not in (0, None)) else np.nan

        freq_diff2 = (f4_mhz - f2_mhz) if (f4_mhz is not None and f2_mhz is not None) else np.nan
        input_data['slope2'] = (rej_f4_db / freq_diff2) if (rej_f4_db is not None and not pd.isna(freq_diff2) and freq_diff2 not in (0, None)) else np.nan

        rejections = [r for r in [rej_f3_db, rej_f4_db] if r is not None]
        input_data['min_rejection'] = min(rejections) if rejections else np.nan
        input_data['max_rejection'] = max(rejections) if rejections else np.nan
        input_data['rej_diff'] = (input_data['max_rejection'] - input_data['min_rejection']) if len(rejections) > 1 else np.nan

        # 2) Impute and scale
        input_df = self._create_feature_dataframe(input_data)
        input_imputed = self.imputer.transform(input_df)
        input_scaled = self.scaler.transform(input_imputed)

        # 3) Predict
        predicted_s21 = float(self.s21_model.predict(input_scaled)[0])
        predicted_s11 = float(self.s11_model.predict(input_scaled)[0])

        return {
            'Predicted S21 (dB)': predicted_s21,
            'Predicted S11 (dB)': predicted_s11,
        }


# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="BPF S-Parameter Predictor", page_icon="📡", layout="wide")

st.title("📡 Bandpass Filter S-Parameter Predictor")
st.caption("Predict S21 (Insertion Loss) and S11 (Return Loss) using trained ML models.")

with st.sidebar:
    st.header("Model Artifacts")
    load_mode = st.radio("Load models via", ["Directory path", "Upload files"], index=0)

    predictor = None
    if 'predictor' not in st.session_state:
        st.session_state['predictor'] = None

    if load_mode == "Directory path":
        default_dir = os.getcwd()
        model_dir = st.text_input("Artifacts directory", value=default_dir, help="Directory containing: bpf_s21_model.pkl, bpf_s11_model.pkl, bpf_scaler.pkl, bpf_imputer.pkl, bpf_features.joblib")
        if st.button("Load from directory", use_container_width=True):
            try:
                st.session_state['predictor'] = BPFPredictor.from_directory(model_dir)
                st.success("Models loaded successfully from directory.")
            except Exception as e:
                st.error(str(e))
    else:
        s21_file = st.file_uploader("bpf_s21_model.pkl", type=["pkl"])  # noqa: F841
        s11_file = st.file_uploader("bpf_s11_model.pkl", type=["pkl"])  # noqa: F841
        scaler_file = st.file_uploader("bpf_scaler.pkl", type=["pkl"])  # noqa: F841
        imputer_file = st.file_uploader("bpf_imputer.pkl", type=["pkl"])  # noqa: F841
        features_file = st.file_uploader("bpf_features.joblib", type=["joblib", "pkl"])  # noqa: F841
        if st.button("Load uploaded files", disabled=not all([s21_file, s11_file, scaler_file, imputer_file, features_file]), use_container_width=True):
            try:
                st.session_state['predictor'] = BPFPredictor.from_uploaded_files(s21_file, s11_file, scaler_file, imputer_file, features_file)
                st.success("Models loaded from uploaded files.")
            except Exception as e:
                st.error(str(e))

    predictor = st.session_state['predictor']

st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Input Parameters")
    unit = st.selectbox("Frequency input unit", ["Hz", "MHz"], index=1)

    c1, c2, c3 = st.columns(3)
    with c1:
        f1_mhz = st.number_input("Passband F1 (MHz)", value=3100.0, min_value=0.0, step=1.0)
    with c2:
        f2_mhz = st.number_input("Passband F2 (MHz)", value=3500.0, min_value=0.0, step=1.0)
    with c3:
        frequency_input = st.number_input(f"Frequency ({unit})", value=3300.0, min_value=0.0, step=1.0)

    z0_ohm = st.number_input("Characteristic Impedance z0 (ohm)", value=50.0, min_value=1.0, step=1.0)

    with st.expander("Optional Stopband Parameters"):
        st.caption("Leave blank if not available")
        c4, c5 = st.columns(2)
        with c4:
            f3_mhz_txt = st.text_input("Stopband F3 (MHz)", value="", placeholder="e.g., 2000")
            rej_f3_db_txt = st.text_input("Rejection @ F3 (dB)", value="", placeholder="e.g., 40")
        with c5:
            f4_mhz_txt = st.text_input("Stopband F4 (MHz)", value="", placeholder="e.g., 4500")
            rej_f4_db_txt = st.text_input("Rejection @ F4 (dB)", value="", placeholder="e.g., 45")

        def _to_float_or_none(s: str) -> Optional[float]:
            s = s.strip()
            if s == "":
                return None
            try:
                return float(s)
            except Exception:
                return None

        f3_mhz = _to_float_or_none(f3_mhz_txt)
        rej_f3_db = _to_float_or_none(rej_f3_db_txt)
        f4_mhz = _to_float_or_none(f4_mhz_txt)
        rej_f4_db = _to_float_or_none(rej_f4_db_txt)

    predict_btn = st.button("Predict S-Parameters", type="primary", use_container_width=True)

with col2:
    st.subheader("Status")
    if predictor and predictor.is_ready():
        st.success("Models ready.")
        st.write(f"Features expected: {len(predictor.features)}")
    else:
        st.info("Load model artifacts to enable predictions.")

# Compute prediction
if predict_btn:
    if not predictor or not predictor.is_ready():
        st.error("Please load the model artifacts first (from directory or upload).")
    else:
        try:
            frequency_hz = frequency_input if unit == "Hz" else frequency_input * 1e6
            result = predictor.predict(
                frequency_hz=frequency_hz,
                f1_mhz=f1_mhz,
                f2_mhz=f2_mhz,
                z0_ohm=z0_ohm,
                f3_mhz=f3_mhz,
                rej_f3_db=rej_f3_db,
                f4_mhz=f4_mhz,
                rej_f4_db=rej_f4_db,
            )

            if 'Error' in result:
                st.error(result['Error'])
            else:
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Predicted S21 (dB)", f"{result['Predicted S21 (dB)']:.4f}")
                with m2:
                    st.metric("Predicted S11 (dB)", f"{result['Predicted S11 (dB)']:.4f}")

                with st.expander("Engineered feature vector preview"):
                    # Recompute the feature frame to preview to the user
                    input_data: Dict[str, Optional[float]] = {
                        'Passband F1 (MHz)': f1_mhz,
                        'Passband F2 (MHz)': f2_mhz,
                        'frequency_hz': frequency_hz,
                        'z0_ohm': z0_ohm,
                        'Stopband F3 (MHz)': f3_mhz,
                        'Rejection @ F3 (dB)': rej_f3_db,
                        'Stopband F4 (MHz)': f4_mhz,
                        'Rejection @ F4 (dB)': rej_f4_db,
                    }
                    f_center = (f1_mhz + f2_mhz) / 2 if (f1_mhz is not None and f2_mhz is not None) else np.nan
                    bw = (f2_mhz - f1_mhz) if (f1_mhz is not None and f2_mhz is not None) else np.nan
                    input_data['f_center'] = f_center
                    input_data['BW'] = bw
                    input_data['frac_BW'] = (bw / f_center) if f_center not in (0, None) and not pd.isna(f_center) else np.nan
                    input_data['Q_approx'] = (f_center / bw) if bw not in (0, None) and not pd.isna(bw) else np.nan
                    input_data['log_f_center'] = np.log10(f_center * 1e6) if f_center is not None and f_center > 0 else np.nan
                    input_data['stop_offset1'] = (f1_mhz - f3_mhz) if (f3_mhz is not None and f1_mhz is not None) else np.nan
                    input_data['stop_offset2'] = (f4_mhz - f2_mhz) if (f4_mhz is not None and f2_mhz is not None) else np.nan
                    input_data['F3_norm'] = (f3_mhz / f_center) if (f3_mhz is not None and f_center not in (0, None) and not pd.isna(f_center)) else np.nan
                    freq_diff1 = (f1_mhz - f3_mhz) if (f3_mhz is not None and f1_mhz is not None) else np.nan
                    input_data['slope1'] = (rej_f3_db / freq_diff1) if (rej_f3_db is not None and not pd.isna(freq_diff1) and freq_diff1 not in (0, None)) else np.nan
                    freq_diff2 = (f4_mhz - f2_mhz) if (f4_mhz is not None and f2_mhz is not None) else np.nan
                    input_data['slope2'] = (rej_f4_db / freq_diff2) if (rej_f4_db is not None and not pd.isna(freq_diff2) and freq_diff2 not in (0, None)) else np.nan
                    rejections = [r for r in [rej_f3_db, rej_f4_db] if r is not None]
                    input_data['min_rejection'] = min(rejections) if rejections else np.nan
                    input_data['max_rejection'] = max(rejections) if rejections else np.nan
                    input_data['rej_diff'] = (input_data['max_rejection'] - input_data['min_rejection']) if len(rejections) > 1 else np.nan
                    preview_df = predictor._create_feature_dataframe(input_data)
                    st.dataframe(preview_df.T.rename(columns={0: 'value'}))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Tip: Ensure your artifacts were saved using the notebook step 5 and copy them next to this app or provide their directory path.")