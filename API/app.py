from flask import Flask, jsonify, request
from flask_restplus import Resource, Api

app = Flask(__name__)
api = Api(app)

@api.route('/')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}
    
if __name__ == '__main__':
    app.run(debug=True)