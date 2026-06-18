import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


# =========================
# 공통 설정
# =========================

DATA_23_PATH = "./HN23_ALL/hn23_all.sas7bdat"
DATA_24_PATH = "./HN24_ALL/hn24_all.sas7bdat"

sns.set_theme(style="whitegrid")

# Windows 한글 폰트 대응
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_and_make_label():
    """2023/2024 국민건강영양조사 SAS 파일을 읽고 대사증후군 라벨을 생성한다."""
    df23 = pd.read_sas(DATA_23_PATH)
    df24 = pd.read_sas(DATA_24_PATH)

    df23["year"] = 2023
    df24["year"] = 2024

    df = pd.concat([df23, df24], ignore_index=True)

    cols = [
        "year",
        "sex",
        "age",
        "HE_wc",
        "HE_sbp",
        "HE_dbp",
        "HE_glu",
        "HE_TG",
        "HE_HDL_st2",
        "HE_BMI",
        "sm_presnt",
        "dr_month",
        "pa_aerobic",
    ]

    df = df[cols].copy()

    for col in cols:
        if col != "year":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()

    # 대사증후군 5가지 기준
    df["abdominal_obesity"] = (
        ((df["sex"] == 1) & (df["HE_wc"] >= 90)) |
        ((df["sex"] == 2) & (df["HE_wc"] >= 85))
    ).astype(int)

    df["high_bp"] = (
        (df["HE_sbp"] >= 130) |
        (df["HE_dbp"] >= 85)
    ).astype(int)

    df["high_glucose"] = (df["HE_glu"] >= 100).astype(int)
    df["high_tg"] = (df["HE_TG"] >= 150).astype(int)

    df["low_hdl"] = (
        ((df["sex"] == 1) & (df["HE_HDL_st2"] < 40)) |
        ((df["sex"] == 2) & (df["HE_HDL_st2"] < 50))
    ).astype(int)

    criteria_cols = [
        "abdominal_obesity",
        "high_bp",
        "high_glucose",
        "high_tg",
        "low_hdl",
    ]

    df["metabolic_syndrome_score"] = df[criteria_cols].sum(axis=1)
    df["metabolic_syndrome"] = (df["metabolic_syndrome_score"] >= 3).astype(int)

    return df


def get_metrics(y_test, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def metrics_to_markdown(metrics_dict):
    lines = ["| Metric | Score |", "|---|---:|"]
    for key, value in metrics_dict.items():
        lines.append(f"| {key} | {value:.4f} |")
    return "\n".join(lines)


def save_target_pie(df, assets_dir):
    counts = df["metabolic_syndrome"].value_counts().sort_index()
    labels = ["정상", "대사증후군"]

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct="%.1f%%", startangle=90)
    plt.title("대사증후군 여부 비율")
    plt.tight_layout()
    path = assets_dir  / "target_ratio_pie.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_histograms(df, columns, assets_dir ):
    paths = []
    for col in columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(data=df, x=col, hue="metabolic_syndrome", kde=True, bins=30)
        plt.title(f"{col} 분포")
        plt.tight_layout()
        path = assets_dir  / f"hist_{col}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def save_boxplots(df, columns, assets_dir ):
    paths = []
    for col in columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=df, x="metabolic_syndrome", y=col)
        plt.title(f"대사증후군 여부에 따른 {col} Boxplot")
        plt.xlabel("metabolic_syndrome (0=정상, 1=대사증후군)")
        plt.tight_layout()
        path = assets_dir  / f"box_{col}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def save_barplots(df, columns, assets_dir ):
    paths = []
    for col in columns:
        rate = df.groupby(col)["metabolic_syndrome"].mean().reset_index()

        plt.figure(figsize=(7, 5))
        sns.barplot(data=rate, x=col, y="metabolic_syndrome")
        plt.title(f"{col}별 대사증후군 비율")
        plt.ylabel("대사증후군 비율")
        plt.tight_layout()
        path = assets_dir  / f"bar_{col}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)
    return paths


def save_correlation_heatmap(df, columns, assets_dir ):
    corr = df[columns + ["metabolic_syndrome"]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("변수 간 상관계수 Heatmap")
    plt.tight_layout()
    path = assets_dir  / "correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_feature_importance(importance_df, assets_dir , filename):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    path = assets_dir  / filename
    plt.savefig(path, dpi=150)
    plt.close()
    return path



def save_roc_curve(y_test, proba_dict, assets_dir):
    plt.figure(figsize=(8, 6))
    for model_name, y_proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    path = assets_dir / "roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_model_comparison(metrics_table, assets_dir):
    plot_df = metrics_table.reset_index().melt(
        id_vars="index",
        var_name="Metric",
        value_name="Score",
    ).rename(columns={"index": "Model"})
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    path = assets_dir / "model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_xgb_feature_importance(importance_df, assets_dir):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    path = assets_dir / "xgb_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def run_chi_square_tests(df, categorical_features):
    results = []

    for feature in categorical_features:
        table = pd.crosstab(df[feature], df["metabolic_syndrome"])
        chi2, p_value, dof, expected = chi2_contingency(table)

        results.append({
            "feature": feature,
            "chi2": chi2,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "table": table,
        })

    return results


def chi_square_to_markdown(results):
    lines = ["| Feature | Chi2 | p-value | 유의 여부 |", "|---|---:|---:|---|"]

    for result in results:
        sig = "유의함" if result["significant"] else "유의하지 않음"
        lines.append(
            f"| {result['feature']} | {result['chi2']:.4f} | {result['p_value']:.6f} | {sig} |"
        )

    return "\n".join(lines)


def confusion_matrix_to_markdown(cm):
    return f"""|  | Pred 0 | Pred 1 |
|---|---:|---:|
| Actual 0 | {cm[0, 0]} | {cm[0, 1]} |
| Actual 1 | {cm[1, 0]} | {cm[1, 1]} |"""


# =========================
# 생활습관 + BMI 기반 모델
# =========================

def main():
    output_dir = Path("./results/lifestyle")
    output_dir.mkdir(parents=True, exist_ok=True)

    assets_dir = output_dir / "assets"
    data_csv_dir = output_dir / "data_csv"
    assets_dir.mkdir(parents=True, exist_ok=True)
    data_csv_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_make_label()

    features = [
        "sex",
        "age",
        "HE_BMI",
        "sm_presnt",
        "dr_month",
        "pa_aerobic",
    ]

    categorical_features = [
        "sex",
        "sm_presnt",
        "dr_month",
        "pa_aerobic",
    ]

    numeric_features = [
        "age",
        "HE_BMI",
    ]

    X = df[features]
    y = df["metabolic_syndrome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    lr_model.fit(X_train_scaled, y_train)

    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = get_metrics(y_test, lr_pred, lr_proba)
    lr_cm = confusion_matrix(y_test, lr_pred)

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": lr_model.coef_[0],
    })
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    coef_df = coef_df.sort_values("odds_ratio", ascending=False)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = get_metrics(y_test, rf_pred, rf_proba)
    rf_cm = confusion_matrix(y_test, rf_pred)

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = get_metrics(y_test, xgb_pred, xgb_proba)
    xgb_cm = confusion_matrix(y_test, xgb_pred)

    xgb_importance_df = pd.DataFrame({
        "feature": features,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    metrics_table = pd.DataFrame({
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
    }).T


    # 5-Fold Cross Validation
    lr_cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        scaler.fit_transform(X),
        y,
        cv=5,
        scoring="roc_auc",
    )
    rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="roc_auc")
    xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring="roc_auc")

    cv_result_df = pd.DataFrame({
        "model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "roc_auc_mean": [lr_cv_scores.mean(), rf_cv_scores.mean(), xgb_cv_scores.mean()],
        "roc_auc_std": [lr_cv_scores.std(), rf_cv_scores.std(), xgb_cv_scores.std()],
    })

    # EDA / 시각화
    target_pie = save_target_pie(df, assets_dir)
    hist_paths = save_histograms(df, numeric_features, assets_dir)
    box_paths = save_boxplots(df, numeric_features, assets_dir)
    bar_paths = save_barplots(df, categorical_features, assets_dir)
    corr_path = save_correlation_heatmap(df, features, assets_dir)
    importance_path = save_feature_importance(
        importance_df,
        assets_dir,
        "rf_feature_importance.png",
    )

    xgb_importance_path = save_xgb_feature_importance(xgb_importance_df, assets_dir)
    roc_path = save_roc_curve(
        y_test,
        {
            "Logistic Regression": lr_proba,
            "Random Forest": rf_proba,
            "XGBoost": xgb_proba,
        },
        assets_dir,
    )
    model_comparison_path = save_model_comparison(metrics_table, assets_dir)


    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        shap_path = assets_dir / "shap_summary.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        shap_path = None
        print(f"SHAP 시각화 생성 실패: {e}")

    hist_md = "".join([f"![{p.stem}](assets/{p.name})" + "\n\n" for p in hist_paths])
    box_md = "".join([f"![{p.stem}](assets/{p.name})" + "\n\n" for p in box_paths])
    bar_md = "".join([f"![{p.stem}](assets/{p.name})" + "\n\n" for p in bar_paths])

    chi_results = run_chi_square_tests(df, categorical_features)

    # 콘솔 출력
    print("\n===== 생활습관 + BMI 기반 모델 =====")
    print(f"데이터 크기: {df.shape}")
    print("\n대사증후군 비율")
    print(df["metabolic_syndrome"].value_counts())
    print(df["metabolic_syndrome"].value_counts(normalize=True).round(4))

    print("\n===== 카이제곱 검정 =====")
    print(chi_square_to_markdown(chi_results))

    print("\n===== Logistic Regression 성능 =====")
    print(metrics_to_markdown(lr_metrics))
    print("\nConfusion Matrix")
    print(lr_cm)
    print("\nOdds Ratio")
    print(coef_df.round(4))

    print("\n===== Random Forest 성능 =====")
    print(metrics_to_markdown(rf_metrics))
    print("\nConfusion Matrix")
    print(rf_cm)
    print("\nFeature Importance")
    print(importance_df.round(4))

    print("\n===== XGBoost 성능 =====")
    print(metrics_to_markdown(xgb_metrics))
    print("\nConfusion Matrix")
    print(xgb_cm)
    print("\nFeature Importance")
    print(xgb_importance_df.round(4))

    print("\n===== 5-Fold Cross Validation ROC-AUC =====")
    print(cv_result_df.round(4))

    # Markdown 보고서 생성
    report = f"""# 생활습관 및 BMI 기반 대사증후군 위험 예측 보고서

## 1. 분석 개요

본 분석은 2023~2024 국민건강영양조사 데이터를 활용하여 대사증후군 위험을 예측하는 것을 목적으로 한다.

이 파일은 건강검진 세부 수치 전체를 사용하지 않고, 비교적 간단히 확보 가능한 생활습관 및 기본 신체지표 중심으로 모델을 구성하였다.

## 2. 데이터 개요

- 전체 분석 대상자 수: {len(df):,}명
- 정상군: {(df["metabolic_syndrome"] == 0).sum():,}명
- 대사증후군군: {(df["metabolic_syndrome"] == 1).sum():,}명
- 대사증후군 비율: {df["metabolic_syndrome"].mean() * 100:.2f}%

![대사증후군 비율](assets/{target_pie.name})

## 3. 사용 변수

| 구분 | 변수 |
|---|---|
| 인구학적 변수 | sex, age |
| 신체지표 | HE_BMI |
| 생활습관 | sm_presnt, dr_month, pa_aerobic |
| 타깃 | metabolic_syndrome |

## 4. 카이제곱 검정 결과

{chi_square_to_markdown(chi_results)}

## 5. 변수 분포 시각화

### 5.1 연속형 변수 Histogram

{hist_md}

### 5.2 대사증후군 여부에 따른 Boxplot

{box_md}

### 5.3 범주형 변수별 대사증후군 비율

{bar_md}

## 6. 상관계수 분석

상관계수는 두 변수 간 선형 관계를 보기 위한 지표이므로 heatmap으로 확인하였다.

![상관계수 Heatmap](assets/{corr_path.name})

## 7. Logistic Regression 결과

### 7.1 성능

{metrics_to_markdown(lr_metrics)}

### 7.2 Confusion Matrix

{confusion_matrix_to_markdown(lr_cm)}

### 7.3 Odds Ratio

{coef_df.round(4).to_markdown(index=False)}

## 8. Random Forest 결과

### 8.1 성능

{metrics_to_markdown(rf_metrics)}

### 8.2 Confusion Matrix

{confusion_matrix_to_markdown(rf_cm)}

### 8.3 Feature Importance

{importance_df.round(4).to_markdown(index=False)}

![Random Forest Feature Importance](assets/{importance_path.name})

## 9. XGBoost 결과

### 9.1 성능

{metrics_to_markdown(xgb_metrics)}

### 9.2 Confusion Matrix

{confusion_matrix_to_markdown(xgb_cm)}

### 9.3 Feature Importance

{xgb_importance_df.round(4).to_markdown(index=False)}

![XGBoost Feature Importance](assets/{xgb_importance_path.name})

## 10. 모델 성능 비교

{metrics_table.round(4).to_markdown()}

![Model Comparison](assets/{model_comparison_path.name})

## 11. ROC Curve

![ROC Curve](assets/{roc_path.name})

## 12. 5-Fold Cross Validation

{cv_result_df.round(4).to_markdown(index=False)}

## 13. SHAP 분석

{"![SHAP Summary](assets/" + shap_path.name + ")" if shap_path is not None else "SHAP 시각화 생성에 실패했습니다."}

## 14. 해석 요약

- 생활습관 및 BMI 기반 모델에서도 대사증후군 위험군을 어느 정도 선별할 수 있었다.
- BMI와 나이가 가장 강한 예측 변수로 나타났다.
- 흡연 여부와 유산소 운동 실천 여부는 카이제곱 검정에서 대사증후군 여부와 유의한 관련성을 보였다.
- 음주 여부는 단순 월간 음주 여부 기준에서는 관련성이 약하게 나타날 수 있다.
- 본 모델은 조기 위험군 선별 목적이므로 Precision보다 Recall과 ROC-AUC를 중요하게 해석하는 것이 적절하다.
"""

    report_path = output_dir / "lifestyle_model_report.md"
    report_path.write_text(report, encoding="utf-8")

    coef_df.to_csv(data_csv_dir / "logistic_odds_ratio.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(data_csv_dir / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")
    xgb_importance_df.to_csv(data_csv_dir / "xgb_feature_importance.csv", index=False, encoding="utf-8-sig")
    metrics_table.to_csv(data_csv_dir / "model_metrics.csv", encoding="utf-8-sig")
    cv_result_df.to_csv(data_csv_dir / "cross_validation_results.csv", index=False, encoding="utf-8-sig")

    print(f"\nMarkdown 보고서 저장 완료: {report_path}")


if __name__ == "__main__":
    main()
