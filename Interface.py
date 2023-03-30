#Creating GUI with tkinter
import tkinter
from tkinter import *
from predict import chatbot_response


base = Tk() #create the window
base.title("PolokBot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE) 

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 14 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

#Create Chat window
#to insert multiline text fields
ChatLog = Text(base, bd=5, bg="#FFD700", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set


#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="8", height=1,
                    bd=5, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox =Text(base, bd=5, bg="white",width="30",height="5", font="Arial")
#EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=124, y=401, height=90, width=265)
SendButton.place(x=5, y=401, height=90)

base.mainloop()
