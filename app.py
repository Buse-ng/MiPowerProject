from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

static_folder = os.path.join(app.root_path, 'static')
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def create_histograms_and_violin_plots(data, num_columns):
    histograms = []
    violin_plots = []

    for col in num_columns:
        # Histogram
        histogram = go.Histogram(
            x=data[col],
            name=col,
            marker=dict(color='rgb(0, 191, 255)') 
        )
        histograms.append(histogram)

        # Violin plot
        violin_plot = go.Violin(
            y=data[col],
            name=col,
            marker=dict(color='rgb(0, 191, 255)'), 
            box_visible=True
        )
        violin_plots.append(violin_plot)

    return histograms, violin_plots


def preprocessing(data):
    data_clean = data.dropna(subset=['ph', 'Sulfate'])
    
    columns_to_fill = ['Trihalomethanes']
    groups = data_clean['Potability'].unique()
    request_list = [] 
    for group in groups:
        group_data = data_clean[data_clean['Potability'] == group]
    
        fill_values = {col: group_data[col].mean() for col in columns_to_fill}
        request_list.append(fill_values)
        data_clean.loc[data_clean['Potability'] == group, columns_to_fill] = group_data[columns_to_fill].fillna(value=fill_values)
    return data_clean, request_list


def anomaly(data):
    req_dict= {}
    for col in data.columns:
        Q3 = np.quantile(data[col], 0.75)
        Q1 = np.quantile(data[col], 0.25)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] > upper_bound) | (data[col] < lower_bound)]
        req_dict[col] = outliers.shape[0]
    return req_dict

def auto_select_target_column(data):
    numerical_columns = data.select_dtypes(include=['number']).columns
    min_variance = float('inf')
    target_column = None
    
    for column in numerical_columns:
        variance = data[column].var()
        if variance < min_variance:
            min_variance = variance
            target_column = column
    
    return target_column

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file.")
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"File saved to {filepath}")
        return redirect(url_for('analyze', filename=file.filename))
    else:
        print("File not uploaded for some reason.")
        return redirect(request.url)


@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    data = pd.read_csv(filepath)

    target_column = auto_select_target_column(data)

    summary = data.describe().to_html()

    cat_columns, num_columns,cat_but_car = grab_col_names(data)

    data_shape = data.shape

    # Box plots
    box_plots = []
    for col in num_columns:
        box_plot = go.Box(
            y=data[col],
            name=col,
            marker=dict(color='rgb(0, 191, 255)')  
        )
        box_plots.append(box_plot)

    plot_div_box_plot = go.Figure(data=box_plots).update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    ).to_html(full_html=False)
    
    histograms, violin_plots = create_histograms_and_violin_plots(data, num_columns)

    # Histograms
    plot_div_histograms = []
    for col, histogram in zip(num_columns,histograms):
        fig = go.Figure(data=[histogram])
        fig.update_layout(
            title={'text': f"{col}", 'x':0.5},
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
        plot_div_histograms.append(fig.to_html(full_html=False))

    # Violin plots
    plot_div_violin_plots = []
    for col, violin_plot in zip(num_columns, violin_plots):
        fig = go.Figure(data=[violin_plot])
        fig.update_layout(
            title={'text': f"{col}", 'x': 0.5},
            xaxis={'visible': False},
            plot_bgcolor='black',
            paper_bgcolor='black',
         font=dict(color='white')
        )
        plot_div_violin_plots.append(fig.to_html(full_html=False))

    cat_analysis = {col: data[col].value_counts() for col in cat_columns}

    # Categorical Analysis
    cat_plots = {}
    for col in cat_columns:
        labels = cat_analysis[col].index.tolist()
        values = cat_analysis[col].values.tolist()

        trace = {
            'x': labels,
            'y': values,
            'type': "bar",
            'name': col,
            'marker': {
                'color': 'rgb(0, 191, 255)' 
                }  
        }

        data1 = [trace]

        #Yatay gridlines eklemek için yaxis ayarları 
        yaxis = {
            'gridcolor' : 'grey', #gridline rengi
            'gridwidth' : 1,
        }

        layout = {
            'title': f'{col}',
            'plot_bgcolor': 'black',
            'paper_bgcolor': 'black',
            'font': {'color': "white"},
            'yaxis': yaxis
        }
    
        cat_plots[col] = {'data': data1, 'layout': layout}

    data_clean, req_list = preprocessing(data)
    anomaly_dict = anomaly(data_clean) 
    
    
    
    return render_template(template_name_or_list= 'analyze.html',
                           filename=filename,
                           tables=[summary],
                           plot_div_box_plot=plot_div_box_plot,
                           plot_div_histograms=plot_div_histograms,
                           plot_div_violin_plots=plot_div_violin_plots,
                           cat_plots=cat_plots,
                           num_columns=num_columns,
                           cat_columns=cat_columns,
                           cat_analysis=cat_analysis,
                           request_list= req_list,
                           data_clean=data_clean,
                           outlier_counts=anomaly_dict,
                           data=data,
                           target_column=target_column)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form
        test_data = {
            'ph': float(form_data['ph']),
            'Hardness': float(form_data['Hardness']),
            'Solids': float(form_data['Solids']),
            'Chloramines': float(form_data['Chloramines']),
            'Sulfate': float(form_data['Sulfate']),
            'Conductivity': float(form_data['Conductivity']),
            'Organic_carbon': float(form_data['Organic_carbon']),
            'Trihalomethanes': float(form_data['Trihalomethanes']),
            'Turbidity': float(form_data['Turbidity'])
        }

        test_df = pd.DataFrame([test_data])
    

        model_path = os.path.join(app.root_path, 'models/random_forest_model.joblib')
        model = joblib.load(model_path)
        
        prediction = model.predict(test_df)
        return render_template('predict.html', prediction=prediction[0])
    else:
        return render_template('predict_form.html')

if __name__ == '__main__':
    app.run(debug=True)

