import sys
import os
from flask import Flask, render_template, request, send_from_directory

# Ensure project root is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.load_data import load_data, get_summary

app = Flask(__name__)


# Custom route to serve generated plots from results folder
@app.route("/results/<filename>")
def serve_results(filename):
    results_dir = os.path.join(project_root, "results")
    return send_from_directory(results_dir, filename)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    try:
        df = load_data()
        summary = get_summary(df)
        # Convert first 5 rows to HTML, styling it using our custom CSS table class
        first_rows_html = df.head().to_html(index=False, classes="dataframe")
        return render_template(
            "load_dataset.html",
            summary=summary,
            first_rows=first_rows_html
        )
    except Exception as e:
        return f"Error loading dataset: {str(e)}", 500


@app.route("/eda")
def eda():
    try:
        df = load_data()
        raw_counts = df["PlacementStatus"].value_counts().to_dict()
        
        # Normalize keys (handles potential float representations e.g. 1.0, 0.0)
        counts = {
            1: int(raw_counts.get(1.0, raw_counts.get(1, 0))),
            0: int(raw_counts.get(0.0, raw_counts.get(0, 0)))
        }
        
        total = counts[1] + counts[0]
        percentages = {
            1: (counts[1] / total * 100) if total > 0 else 0,
            0: (counts[0] / total * 100) if total > 0 else 0
        }
        
        return render_template(
            "eda_page.html",
            counts=counts,
            percentages=percentages,
            total_features=len(df.columns)
        )
    except Exception as e:
        return f"Error running exploratory data analysis: {str(e)}", 500


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            cgpa = float(request.form.get("cgpa", 0))
            internships = int(request.form.get("internships", 0))
            projects = int(request.form.get("projects", 0))
            backlogs = int(request.form.get("backlogs", 0))
            extracurricular = int(request.form.get("extracurricular", 0))
            communication = int(request.form.get("communication", 3))

            # Heuristic calculation for Placement Probability
            # CGPA: max 45% (cgpa of 10 gives 0.45)
            score = (cgpa / 10.0) * 0.45
            
            # Internships: max 20% (2 or more gives 0.20)
            score += min(internships, 2) * 0.10
            
            # Projects: max 15% (3 or more gives 0.15)
            score += min(projects, 3) * 0.05
            
            # Extracurricular activities: max 10%
            score += 0.10 if extracurricular == 1 else 0.02
            
            # Communication skills: max 10% (Excellent=0.10, Good=0.07, Average=0.04)
            if communication == 3:
                score += 0.10
            elif communication == 2:
                score += 0.07
            else:
                score += 0.04

            # Backlogs penalty: reduces score by 15% per backlog, max 30% reduction
            score -= min(backlogs, 2) * 0.15

            # Clamp the probability between 0.05 and 0.98
            probability = max(0.05, min(0.98, score))
            
            # Decide prediction based on 50% threshold and backlog check
            prediction_val = 1 if (probability >= 0.50 and backlogs == 0) or (probability >= 0.65 and backlogs <= 1) else 0
            
            return render_template(
                "prediction.html", 
                prediction=prediction_val, 
                probability=probability,
                cgpa_val=cgpa,
                internships_val=internships,
                projects_val=projects,
                backlogs_val=backlogs,
                extracurricular_val=extracurricular,
                communication_val=communication
            )
        except Exception as e:
            return render_template("prediction.html", error=str(e))
            
    return render_template("prediction.html")


# Placeholder routes for future modules
@app.route("/preprocessing")
def preprocessing():
    return render_template(
        "placeholder.html",
        title="Data Preprocessing",
        icon="fa-solid fa-gears",
        description="This section will contain pipelines to clean the data, handle missing features, scale numerical attributes, encode categorical strings, and divide the dataset into training and validation sets for model engineering."
    )


@app.route("/models")
def models():
    return render_template(
        "placeholder.html",
        title="Model Engineering",
        icon="fa-solid fa-brain",
        description="This section will train and compare multiple machine learning classifiers (such as Logistic Regression, Decision Trees, Random Forests, and SVMs) using Scikit-Learn to identify the optimal model."
    )


@app.route("/evaluation")
def evaluation():
    return render_template(
        "placeholder.html",
        title="Model Evaluation",
        icon="fa-solid fa-square-check",
        description="This section will evaluate performance metrics of the trained models, analyzing details like Accuracy, Precision, Recall, F1-Scores, and the Confusion Matrix."
    )


@app.route("/comparison")
def comparison():
    return render_template(
        "placeholder.html",
        title="Model Comparison",
        icon="fa-solid fa-chart-line",
        description="This section will showcase comparative performance graphs and ROC curves of different classification techniques to justify the final model selection."
    )


if __name__ == "__main__":
    app.run(debug=True)