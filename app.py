import logging

from flask import Flask, request, render_template, jsonify
from flask.logging import default_handler

from server import TradingServer, InvalidInputError

app = Flask(__name__.split('.')[0])

logger = logging.getLogger()
logger.addHandler(default_handler)

class ResultsManager():
    """
    A simple class to handle flask operations for the model.

    Attributes
    ---
    server : class TradingServer
        The top-level manager process for getting actions for the model and
        updating the user's state
    data : list
        A list of the history for the user based on the model's actions

    Methods
    ---
    get_results : JSON-serialized dict
        Processes user input and updates state for POST requests
    print_results : flask.template
        Displays state for GET requests 
    """


    def __init__(self):
        self.server = TradingServer()
        self.data = []

    def get_results(self, user_input: dict):
        """
        Processes user input and updates state for POST requests.
        
        Parameters
        ---
        user_input : dict
            A dict of the user input

        Returns
        ---
        A JSON-serialized dict of the user's state and the model's actions
        """
        try:
            state_for_client = self.server.process_user_input(user_input)
            self.data.append(list(state_for_client.values()))
        except InvalidInputError as e:
            message = e.message
            state_for_client = {'Invalid Input Error' : message}
        return jsonify(state_for_client)

    def print_results(self):
        """
        Displays state for GET requests.

        Returns
        ---
        An HTML template filled with the user's history of actions and state
        """
        return render_template('table.html', data=self.data)

@app.route('/api/results', methods=['GET', 'POST'])
def handle_user_input():
    if request.method == 'POST':
        user_input = request.get_json(force=True, cache=True)
        return mgr.get_results(user_input)
    if request.method == 'GET':
        return mgr.print_results()

if __name__ == '__main__':
    mgr = ResultsManager()
    app.run(host='0.0.0.0',debug=True)
