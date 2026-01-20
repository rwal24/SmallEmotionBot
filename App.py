import model_training as mt
from flask import Flask, redirect, url_for, render_template, request
# to run, use command: python -m http.server
# whichever port to run on depends on what u want

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", attempt=False, success=False)


# to display when the model is being trained
@app.route('/submit', methods=["POST"])
def submit():
    text_input = request.form.get('input')
    selected_emotion = request.form.get('emotion')
    
    emotion_arr = [
                    "Happy",
                    "Sad",
                    "Angry",
                    "Confused",
                    "Frustrated",
                    "Scared",
                    "Surpised",
                    "Disgusted",
                    "Anxious",
                    "Shame",
                    "Excited"
                ]
    
    for i in range(len(emotion_arr)):
        if emotion_arr[i] == selected_emotion:
            success = mt.main(text_input, i, 1)
            if success == 1:
                return render_template("index.html", attempt=True, success=False)

            if success == 0:
                return render_template("index.html", attempt=True, success=True)
            

    return render_template("index.html", attempt=True, success=False)





# to display when the model is being tested
@app.route('/test', methods=["GET", "POST"])
def test():
    if request.method == "POST":
        text_input = request.form.get('input')
        # 0 used in place of emotion_int, since in this case it will not be used
        probabilities = mt.main(text_input, 0, 2)

        if not isinstance(probabilities, list):
            return render_template("test.html", attempt=True, success=False, probabilities=probabilities, LargestEmotion="None")

        else:
            max_prob = 0
            max_emotion = ""
            for item in probabilities:
                if item[1] > max_prob:
                    max_prob = item[1]
                    max_emotion = item[0]

            return render_template("test.html", attempt=True, success=True, probabilities=probabilities, LargestEmotion=max_emotion)
        

    else:
        return render_template("test.html", attempt=False, success=False, probabilitiesb=[], LargestEmotion="None")
    






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)