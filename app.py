from flask import Flask, render_template, url_for, request
import numpy as np

import pickle
import csv
from tqdm import tqdm
from scipy import sparse
tqdm.pandas(desc="progress-bar")

app = Flask(__name__)


@app.route('/', methods=['POST','GET'])
def home():

    if request.method == 'POST':

        tags = ['From the Center', 'From the Left', 'From the Right']

        left_prediction = F"{request.form['left_prediction']}"
        right_prediction = F"{request.form['right_prediction']}"
        center_prediction = F"{request.form['center_prediction']}"
        prediction = F"{request.form['prediction']}"

        return render_template('result.html',
                               left_prediction=left_prediction,
                               right_prediction=right_prediction,
                               center_prediction=center_prediction,
                               prediction=prediction)

    return render_template('front.html',
                           left_prediction="0.00%",
                           right_prediction="0.00%",
                           center_prediction="0.00%",
                           prediction="No predictions yet! Enter an article!")


@app.route('/results', methods=['POST', 'GET'])
def predict():

    if request.method == 'POST':
        message = request.form['article']
        data = [message]

        text_counts = pre_process(data)
        prediction = classify(text_counts)

        tags = ['From the Center', 'From the Left', 'From the Right']

        left_prediction = F"{prediction[0][1]:.2%}"
        right_prediction = F"{prediction[0][2]:.2%}"
        center_prediction = F"{prediction[0][0]:.2%}"

        best = max(prediction[0][1], prediction[0][2], prediction[0][0])
        prediction = tags[np.where(prediction[0] == best)[0][0]]

        return render_template('result.html',
                               left_prediction=left_prediction,
                               right_prediction=right_prediction,
                               center_prediction=center_prediction,
                               prediction=prediction)

    return render_template('result.html',
                           left_prediction="0.00%",
                           right_prediction="0.00%",
                           center_prediction="0.00%",
                           prediction="No predictions yet! Enter an article!")


def pre_process(data):

    LEN = len(data)

    filename = F"datasets/ibc_cv.sav"
    cv = pickle.load(open(filename, 'rb'))

    print("\nGenerate bag of words matrix...")
    text_counts = cv.transform(data)  # returns a sparse matrix, entry = matrix[x, y]

    filename = 'datasets/feature_dict.sav'
    feature_dict = pickle.load(open(filename, 'rb'))

    NEU_LEN, LIB_LEN, CON_LEN = 11040, 3272, 3272
    ROW_LEN = LEN
    FACTORS = (0.01, .0025, .0015)

    print('\nIntegrating IBC data...')

    def integrate_ibc(path, length, tc):
        lil_tc = sparse.lil_matrix(tc)
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_of_words = int(row['gram'])
                if num_of_words == 1:
                    word = row['1st']
                elif num_of_words == 2:
                    word = F"{row['1st']} {row['2nd']}"
                elif num_of_words == 3:
                    word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                if word in feature_dict:
                    word_i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if lil_tc[doc_i, word_i] > 0:
                            lil_tc[doc_i, word_i] *= float(row['freq']) / FACTORS[num_of_words - 1]

            return sparse.csr_matrix(lil_tc)

    text_counts = integrate_ibc("datasets/neu_list.csv", NEU_LEN, text_counts)
    text_counts = integrate_ibc("datasets/lib_list.csv", LIB_LEN, text_counts)
    text_counts = integrate_ibc("datasets/con_list.csv", CON_LEN, text_counts)

    return text_counts


def classify(text_counts):
    CLF_NAME = "AdaBoostClassifier"
    PERCENT = "93.956%"

    filename = F"datasets/{PERCENT}_ibc_{CLF_NAME}.sav"
    clf = pickle.load(open(filename, 'rb'))

    return clf.predict_proba(text_counts)


if __name__ == '__main__':
    app.run(debug=True)
