from flask import Flask, render_template, request
from transformers import AutoTokenizer
import pickle

app = Flask(__name__)

models =[]
name =[]
tokrnizers =[]

model1 = pickle.load(open("t5-small.pkl","rb"))
model2 = pickle.load(open("t5-small-adapter.pkl","rb"))
# model3 = pickle.load(open("t5-base.pkl","rb"))
# model4 = pickle.load(open("t5-base-adapter.pkl","rb"))

tokenizer1 = AutoTokenizer.from_pretrained('t5-small')
# tokenizer2 = AutoTokenizer.from_pretrained('t5-base')

name.append('T5-small-finetuned(CNNDM)')
models.append(model1)
tokrnizers.append(tokenizer1)

name.append('T5-small-adapter(CNNDM)')
models.append(model2)
tokrnizers.append(tokenizer1)

# name.append('T5-base-finetuned(CNNDM)')
# models.append(model3)
# tokrnizers.append(tokenizer2)

# name.append('T5-base-adapter(CNNDM)')
# models.append(model4)
# tokrnizers.append(tokenizer2)

# ===========================================================================================================================
# ===========================================================================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
    if request.method == 'POST':
        txt =  request.form['txt']
        txt = 'summarize: ' + txt
        results = []
        
        for model,tokenizer in zip(models,tokrnizers):
            inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True)
            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=True,  # disable sampling to test if batching affects output,
                min_length=30,
                max_length=150,
                # early_stopping=True
            )

            final_output = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            results.append(final_output)

        info = zip(name,results)
        return render_template('output.html', info=info,text=txt)
    else:
        return render_template('index.html')

    

if __name__ == "__main__":
    app.run(debug=True)
