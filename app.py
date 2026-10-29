import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ShopperSense",
    page_icon="🛍️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "shopper_models"
DATA_PATH = BASE_DIR / "online_shoppers_intention.csv"

st.markdown("""
<style>
.stApp {
    background-color: #F3F7F6;
}
h1, h2, h3 {
    color: #087F70;
}
[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD SAVED FILES ----------------
@st.cache_resource
def load_model(filename):
    path = MODEL_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


models = {
    "Random Forest": load_model("random_forest.joblib"),
    "Decision Tree": load_model("decision_tree.joblib"),
    "SVM": load_model("svm.joblib"),
    "Logistic Regression": load_model("logistic_regression.joblib"),
    "KNN": load_model("knn.joblib"),
}

df = load_dataset()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛍️ ShopperSense")
st.sidebar.caption("Online Shopping Purchase Intention")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Model Comparison", "Purchase Prediction",
     "Customer Segments", "Dataset Explorer"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("🛍️ ShopperSense")
    st.subheader("Online Shopper Purchase Intention Analytics")
    st.write(
        "An interactive machine-learning application that analyses "
        "online shopping session behaviour and estimates purchase intention."
    )

    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset Records", f"{len(df):,}")
        c2.metric("Input Features", str(df.shape[1] - 1))
        c3.metric(
            "Purchase Sessions",
            f"{int(df['Revenue'].astype(bool).sum()):,}"
        )

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Purchase Distribution")
        counts = df["Revenue"].astype(str).value_counts()
        st.bar_chart(counts)
    else:
        st.warning(
            "Dataset not found. Place online_shoppers_intention.csv "
            "in the main ShopperSense folder."
        )

    st.markdown("### Project workflow")
    st.write(
        "Data cleaning → preprocessing → model tuning and evaluation "
        "→ purchase prediction → customer segmentation"
    )


# ---------------- MODEL COMPARISON ----------------
elif page == "Model Comparison":
    st.title("Model Performance Comparison")

    results_path = MODEL_DIR / "model_comparison.csv"

    if results_path.exists():
        results = pd.read_csv(results_path)
        st.dataframe(results, use_container_width=True)

        # Display a chart if metric columns can be identified.
        lower_cols = {c.lower().replace("_", " "): c for c in results.columns}
        model_col = next(
            (c for c in results.columns
             if "model" in c.lower() or "algorithm" in c.lower()),
            results.columns[0]
        )

        metric_candidates = []
        for wanted in ["f1", "accuracy", "precision", "recall", "roc auc"]:
            match = next(
                (actual for normal, actual in lower_cols.items()
                 if wanted in normal),
                None
            )
            if match and match != model_col:
                metric_candidates.append(match)

        if metric_candidates:
            chart_data = results.set_index(model_col)[metric_candidates]
            st.markdown("### Metric Comparison")
            st.bar_chart(chart_data)

    else:
        st.error("model_comparison.csv was not found in shopper_models.")


# ---------------- PURCHASE PREDICTION ----------------
elif page == "Purchase Prediction":
    st.title("Predict Purchase Intention")
    st.write(
        "Enter details about a website session. Choose a saved model "
        "to estimate whether the visitor is likely to make a purchase."
    )

    available_models = {
        name: model for name, model in models.items()
        if model is not None
    }

    if not available_models:
        st.error("No saved prediction models were found.")
        st.stop()

    selected_model = st.selectbox(
        "Select a model", list(available_models.keys())
    )

    with st.form("prediction_form"):
        st.markdown("#### Page activity")
        a, b, c = st.columns(3)

        with a:
            administrative = st.number_input(
                "Administrative pages", min_value=0, value=0
            )
            informational = st.number_input(
                "Informational pages", min_value=0, value=0
            )
            product_related = st.number_input(
                "Product-related pages", min_value=0, value=5
            )

        with b:
            admin_duration = st.number_input(
                "Administrative duration", min_value=0.0, value=0.0
            )
            info_duration = st.number_input(
                "Informational duration", min_value=0.0, value=0.0
            )
            product_duration = st.number_input(
                "Product-related duration", min_value=0.0, value=100.0
            )

        with c:
            bounce = st.number_input(
                "Bounce rate", min_value=0.0, max_value=1.0, value=0.02
            )
            exit_rate = st.number_input(
                "Exit rate", min_value=0.0, max_value=1.0, value=0.04
            )
            page_values = st.number_input(
                "Page value", min_value=0.0, value=0.0
            )

        st.markdown("#### Visitor and session details")
        d, e, f = st.columns(3)

        with d:
            special_day = st.number_input(
                "Special day", min_value=0.0, max_value=1.0, value=0.0
            )
            operating_system = st.number_input(
                "Operating system code", min_value=1, value=2
            )
            browser = st.number_input(
                "Browser code", min_value=1, value=2
            )

        with e:
            region = st.number_input("Region code", min_value=1, value=1)
            traffic_type = st.number_input(
                "Traffic type code", min_value=1, value=2
            )
            month = st.selectbox(
                "Month",
                ["Feb", "Mar", "May", "June", "Jul", "Aug",
                 "Sep", "Oct", "Nov", "Dec"]
            )

        with f:
            visitor_type = st.selectbox(
                "Visitor type",
                ["Returning_Visitor", "New_Visitor", "Other"]
            )
            weekend = st.selectbox("Weekend visit?", [False, True])

        submitted = st.form_submit_button(
            "Predict Purchase Intention",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": admin_duration,
            "Informational": informational,
            "Informational_Duration": info_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce,
            "ExitRates": exit_rate,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }])

        model = available_models[selected_model]

        try:
            prediction = int(model.predict(input_data)[0])

            st.markdown("### Prediction Result")
            if prediction == 1:
                st.success("The model predicts that the visitor is likely to purchase.")
            else:
                st.info("The model predicts that the visitor is unlikely to purchase.")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)
                if 1 in classes:
                    purchase_probability = probabilities[classes.index(1)]
                    st.metric(
                        "Estimated purchase probability",
                        f"{purchase_probability:.1%}"
                    )
                    st.caption(
                        "This is the model's estimated probability, "
                        "not a guarantee of a purchase."
                    )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.caption(
                "Check that the saved model is a complete preprocessing "
                "and prediction pipeline."
            )


# ---------------- CUSTOMER SEGMENTS ----------------
elif page == "Customer Segments":
    st.title("Customer Segmentation")
    st.write(
        "Explore K-Means clusters formed from online shopping session "
        "behaviour. Clusters represent behavioural groups, not guaranteed "
        "customer identities."
    )

    kmeans = load_model("kmeans_model.joblib")
    cluster_preprocessor = load_model("cluster_preprocessor.joblib")
    pca = load_model("kmeans_visualization_pca.joblib")

    if df is None:
        st.warning("Place the dataset CSV in the main project folder.")
    elif kmeans is None or cluster_preprocessor is None:
        st.error("The saved K-Means model or preprocessor is missing.")
    else:
        try:
            features = df.drop(columns=["Revenue"], errors="ignore")
            processed = cluster_preprocessor.transform(features)
            labels = kmeans.predict(processed)

            segment_df = pd.DataFrame({
                "Cluster": labels.astype(str)
            })

            st.markdown("### Cluster Sizes")
            st.bar_chart(segment_df["Cluster"].value_counts().sort_index())

            if pca is not None:
                coordinates = pca.transform(processed)
                plot_df = pd.DataFrame({
                    "Component 1": coordinates[:, 0],
                    "Component 2": coordinates[:, 1],
                    "Cluster": labels.astype(str)
                })

                st.markdown("### Cluster Visualisation")
                fig, ax = plt.subplots()
                for cluster in sorted(plot_df["Cluster"].unique()):
                    subset = plot_df[plot_df["Cluster"] == cluster]
                    ax.scatter(
                        subset["Component 1"],
                        subset["Component 2"],
                        label=f"Cluster {cluster}",
                        alpha=0.6,
                        s=15
                    )
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            st.caption(
                "This page assigns sessions to the saved K-Means model. "
                "PCA is used only to display a two-dimensional view."
            )

        except Exception as error:
            st.error(f"Could not display clusters: {error}")
            st.caption(
                "A preprocessing or scikit-learn version mismatch may "
                "need to be resolved."
            )


# ---------------- DATASET EXPLORER ----------------
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")

    if df is None:
        st.error("Dataset file not found.")
    else:
        st.markdown("### Dataset Information")
        st.write(f"Rows: {df.shape[0]:,}")
        st.write(f"Columns: {df.shape[1]}")
        st.write(f"Duplicate rows: {int(df.duplicated().sum()):,}")

        st.markdown("### Missing Values")
        missing = df.isnull().sum().rename("Missing values")
        st.dataframe(missing.to_frame(), use_container_width=True)

        st.markdown("### Explore Data")
        selected_column = st.selectbox(
            "Choose a column", df.columns.tolist()
        )
        st.write(df[selected_column].value_counts(dropna=False).head(20))

        st.markdown("### Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
