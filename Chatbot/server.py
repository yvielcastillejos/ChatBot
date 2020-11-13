from flask import Flask,request, url_for, redirect, render_template
from flask_socketio import SocketIO, emit, send
import pandas as pd
from bot import*

app = Flask(__name__)
app.config[ 'SECRET_KEY' ] = 'mysecret'
socketio = SocketIO(app, cors_allowed_origins= "*")

path =  "/Users/yvielcastillejos/python_code/Chatbot"
answe = ""

def loadmodels(TEXT_I, path):
      cnn = CNN_net(TEXT_I.vocab, 50,[1,1])
      #rnn = RNN_net(TEXT_I.vocab)
  
      cnn = torch.load(f'{path}/model_cnn.pt', map_location=torch.device    ('cpu'))
      #rnn = torch.load(f'{path}model_rnn.pt')
  
      cnn.eval()
      #rnn.eval()
      return  cnn 






@app.route('/')
def hello_world():
    return render_template("chat.html")



@socketio.on( 'message' )
def handleMessage(msg):
    print('Message:'+ msg)
    # We are not broadcasting because we are not sending it to another client; It will default to sending the message to whoever sent it
    # This is a chat app
    df = pd.DataFrame()
    data2 = dict()
    cnn = loadmodels(TEXT, path)
    # Gives out a dictionary of the user input
    sentence = str(request.form.values())
    to_see = dict()
    label = 0
    text = msg
    to_see[text] = label
    df2 = pd.DataFrame(list(to_see.items()), columns = ['text', 'label'])
    df2.to_csv(f"{path}/user.tsv", sep='\t', index=False)
    example = [msg]
    user_iter = Tokenize2(path)
    for i, data in enumerate(user_iter, 0):
        if msg != '' or msg != None:
            print(f"msg is {msg}")
            batch_input, batch_input_length = data.text
            batch_labels = (data.label)

            cnn_out = cnn(batch_input, batch_input_length)
            cnn_ans = convert(cnn_out)
            cnn_ans_class = get_class(cnn_out)
            response_list = responses[cnn_ans_class]
            response =  random.choice(response_list)
    if msg == None or msg == "":
        response = answe
    msg2 = dict()
    msg2 = {'message':msg, 'answer': response}
    send(msg2, broadcast = True)

if __name__ == '__main__':
  # wraps around the socket io
    # Flask app waits for request respons
    # Socket iois a real time 
    # adding to the standard server when running app and 
    # have a real time functionality
  socketio.run( app, debug = True )

'''
@app.route('/bot',methods=['POST','GET'])
def bot():
    userText = request.args.get("msg")
    response = str(request.form.values())
    print(response)

    df = pd.DataFrame()
    data2 = dict()
    cnn = loadmodels(TEXT, path)
    # Gives out a dictionary of the user input
    sentence = str(request.form.values())
    to_see = dict()
    label = 0
    text = sentence
    to_see[text] = label
    df2 = pd.DataFrame(list(to_see.items()), columns = ['text', 'label'])
    df2.to_csv(f"{path}/user.tsv", sep='\t', index=False)
    example = [sentence]
    user_iter = Tokenize2(path)
    for i, data in enumerate(user_iter, 0):
        batch_input, batch_input_length = data.text
        batch_labels = (data.label)

        cnn_out = cnn(batch_input, batch_input_length)
        cnn_ans = convert(cnn_out)
        cnn_ans_class = get_class(cnn_out)
        response_list = responses[cnn_ans_class]
        response =  random.choice(response_list)
    return render_template('chat.html') #, pred='{}'.format(response))
    
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
if __name__ == '__main__':
    app.run(debug=True)'''
