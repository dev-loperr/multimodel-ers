from flask import Flask, request
app = Flask(__name__)
app.config['MongoDB_Connection_String'] = ''


@app.route('/', methods=['POST', 'GET'])
def index():
    return "ERS-Backend is up!"

if __name__ == "__main__":
    app.run(debug=True)