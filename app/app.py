#This is the main application handler for Hippocrates experiment creation
#it provides functionality for its main route /experiment
#Arkangel AI
#Responsible: Nicolas Munera

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS,cross_origin
from six.moves import http_client
from api_functions import *
import os

app_IP = '0.0.0.0'
app= Flask(__name__)
CORS(app, support_credentials=True)

#0. Disable flask pretify for json optimization
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['JSON_SORT_KEYS'] = False
app.config['UPLOAD_FOLDER'] = os.getcwd()
print(os.getcwd())

@app.route('/leukemia_predict', methods=['POST'])
def leukemia_predict():
    """Simple echo service."""
    json_data = request.get_json()
    res_dict = api_predict(json_data)
    return jsonify(res_dict)
    try:
        json_data = request.get_json()
        res_dict = api_predict(json_data)
    except Exception as e:
        print('error: ', e)
        return jsonify( { 'error':str(e),  'status':'error'} )
    return jsonify(res_dict)


@app.route("/uploads/<path:name>")
def download_file(name):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'], name, as_attachment=True
    )

@app.route("/health_check", methods=['GET'])
def health_check():
    return jsonify( {'respuesta': 'estoy vivo'} )


@app.route('/auth/info/googlejwt', methods=['GET'])
def auth_info_google_jwt():
    """Auth info with Google signed JWT."""
    return auth_info()

@app.route('/auth/info/googleidtoken', methods=['GET'])
def auth_info_google_id_token():
    """Auth info with Google ID token."""
    return auth_info()


@app.route('/auth/info/firebase', methods=['GET'])
@cross_origin(send_wildcard=True)
def auth_info_firebase():
    """Auth info with Firebase auth."""
    return auth_info()

@app.errorhandler(http_client.INTERNAL_SERVER_ERROR)
def unexpected_error(e):
    """Handle exceptions by returning swagger-compliant json."""
    logging.exception('An error occured while processing the request.')
    response = jsonify({
        'code': http_client.INTERNAL_SERVER_ERROR,
        'message': 'Exception: {}'.format(e)})
    response.status_code = http_client.INTERNAL_SERVER_ERROR
    return response

if __name__ == '__main__':
    app.run(host=app_IP, port=1056)
