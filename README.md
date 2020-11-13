# ChatBot
Here, I trained a Chatbot using Natural Language Processing and Neural Networks. I used the Spacy Library to tokenize the words and I vectorized them with CNNs. So far, the chat bot is very limited, which I owe to the lack of data. I learned quite a lot from this project (Besides implementing NLP to build a chatbot, I learned a little bit of how javascript and Flask work-- some keyterms I learned include http, CORS, sockets, clients, servers, broadcasting to a server, POST, GET, socket emit and send, etc..)

## NLP Process

- In the NLP Process, I basically implemented what I learned in one of my assignments for one of my courses. For this one, I decided to use a CNN to train the model. Using a CNN for NLP was first proposed by Yoon (Yoon Kim. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751. Association for Computational Linguistics, 2014.)

- Furthermore, The CNN uses kernel size of 1x1 and uses a Relu Activation Function with one hidden Fully Connected Layer. In order to handle the changing lengths of the sentences, a MaxPool operation on the convolution layer output (after the words have been vectorized) was performed (along the sentence length dimension). It computes the maximum across the entire sentence length, and gets one output feature/number from each sentence for each kernel. 

- Finally, it classifies the sentence into one of the semantic topics. This classification was used to arrive to a final response.

## The Chatbot

- I created a website that allows me to chat with the chatbot. I hosted a server using javascript and designed by website using HTML and CSS. Furthermore, I used Flask as a framework in order to deploy my CNN model. Flask is a framework that allows one to integrate python into webdev tools. I watched quite a lot of tutorials on how it works and in the process, learned a little bit about how servers work.

## Demo

- I acknowledge that my bot does not perform that well because of the lack of data. However, I am confident that if I were to increase my dataset, it would be really good. (It reached 100% training accuracy in around 5 epochs with my current dataset; so it is very promising)
- As I spent a day on this project (and same with most of my projects here on github) I am not commited enough to increase my dataset. However, I might choose to make this project better in the future.

## Future Steps
- I originally planned this to be a "mental health check" type of bot or maybe even a therapist bot. For this to work, I really need a good dataset and a bunch more to mimic actual conversations. 
- I can easily make this chatbot into customer service bots, Question and Answer bots, etc.
- I would like to add another topic, which is "Questions", and the bot would search up the internet for answers. This is pretty doable given that there is a library that makes python scrape Google. Furthermore, the bot will be able to classify questions no problem (it just looks for words startingw ith W5H most of the time).

## Acknowledgement
[1]  https://www.youtube.com/watch?v=lU8-Bs-iXNM A tutorial on how to make a basic website with chat (Note: Does not use Flask as this tutorial does not teach how to deploy a NN model)
[2] https://www.youtube.com/watch?v=9MHYHgh4jYc Good Lesson on a little bit of how flask works and POST, and GET
[3] https://www.youtube.com/watch?v=Pc8WdnIdXZg&t=512s Gave me the idea to use flask to deploy model
[4] https://www.youtube.com/watch?v=lU8-Bs-iXNM&t=326s Good Lesson on Flask and its limitations and web socket, socket io, client servers, etc.