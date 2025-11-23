# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from flask import Flask

app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hola mundo by Edu'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
