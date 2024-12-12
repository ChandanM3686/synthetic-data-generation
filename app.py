import pandas as pd
import os
import logging
from ctgan import CTGAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from flask import Flask, render_template, request, redirect, url_for, send_file, flash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'secret_key'

logging.basicConfig(level=logging.DEBUG)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def upload_page():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_synthetic_data():
    if 'file' not in request.files:
        flash("No file uploaded!")
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    if file.filename == '':
        flash("No file selected!")
        return redirect(url_for('upload_page'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    try:
        logging.info("Saving file to: %s", file_path)
        file.save(file_path)

        logging.info("Loading CSV file.")
        df = pd.read_csv(file_path)
        logging.info("CSV loaded successfully. Columns: %s", df.columns)

        if df.isnull().values.any():
            logging.info("Missing values detected in the dataset.")
            imputer = SimpleImputer(strategy="most_frequent")
            df[:] = imputer.fit_transform(df)
            logging.info("Missing values imputed using 'most_frequent' strategy.")

        categorical_features = request.form.get('categorical_features', '').strip()
        if categorical_features:
            categorical_features = [col.strip() for col in categorical_features.split(',')]
        else:
            categorical_features = []
        
        logging.info("Categorical features: %s", categorical_features)

        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)

        label_encoders = {}
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
                logging.info(f"Encoded categorical column: {col}")

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logging.warning(f"Could not convert column {col}: {e}")
                    df[col] = df[col].fillna(0)
        
        logging.info("Starting CTGAN model training.")
        ctgan = CTGAN(verbose=True)
        ctgan.fit(df, categorical_features, epochs=200)
        logging.info("CTGAN model trained successfully.")

        samples = ctgan.sample(1000)
        logging.info("Synthetic data generated.")

        missing_cols = set(df.columns) - set(samples.columns)
        extra_cols = set(samples.columns) - set(df.columns)

        for col in missing_cols:
            samples[col] = pd.NA
        for col in extra_cols:
            samples.drop(columns=[col], inplace=True)

        samples = samples[df.columns]

        synthetic_file_name = f"synthetic_{os.path.splitext(file.filename)[0]}.csv"
        synthetic_file_path = os.path.join(app.config['UPLOAD_FOLDER'], synthetic_file_name)
        samples.to_csv(synthetic_file_path, index=False)
        logging.info("Synthetic data saved to: %s", synthetic_file_path)

        plot_comparison(df, samples)

        ks_results = calculate_ks_test(df, samples)

        return render_template('result.html', synthetic_file_path=synthetic_file_name, ks_results=ks_results)
    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        flash(f"Error processing file: {e}")
        return redirect(url_for('upload_page'))

def plot_comparison(original_data, synthetic_data):
    numeric_columns = original_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(original_data[col], label='Original Data', color='blue')
        sns.kdeplot(synthetic_data[col], label='Synthetic Data', color='red')
        plt.title(f"Comparison of {col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f"{col}_comparison.png"))
        plt.close()

def calculate_ks_test(original_data, synthetic_data):
    ks_results = {}
    numeric_columns = original_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        stat, p_value = ks_2samp(original_data[col].dropna(), synthetic_data[col].dropna())
        ks_results[col] = {"statistic": stat, "p_value": p_value}
    return ks_results

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash("File not found!")
        return redirect(url_for('upload_page'))
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
