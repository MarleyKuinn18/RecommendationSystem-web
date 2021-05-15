import uvicorn
from fastapi import FastAPI
from model import graphVisualization

app = FastAPI()
model = graphVisualization()

@app.get('/')
def index():
    return {'message': 'hello'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

@app.post('/draw')
def draw_network():
    network = model.simple_graph()
    return {
        'network': network
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

