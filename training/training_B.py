from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
from sklearn.metrics import classification_report

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=343)

def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results

def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id    # put your model here
    )

    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)


    trainer.save_model(best_model_path)
    
def test(test_df, model_path, id2label, label2id):

    # load tokenizer from saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)

    # return dictionary of classification report
    return results, preds

# define input arguments
train_path = "subtaskB_train.jsonl"
test_path =  "subtaskB_dev.jsonl"
#model_name = "google/bert_uncased_L-8_H-512_A-8" # monolilingual model
model_name =  'roberta-base' # monolilingual model (0.65 mF1)
#model_name =  'xlm-roberta-base' # multilingual model
subtask =  "B"
prediction_path = "subtask"+subtask+"_predictions.jsonl"

random_seed = 0

if not os.path.exists(train_path):
  logging.error("File doesnt exists: {}".format(train_path))
  raise ValueError("File doesnt exists: {}".format(train_path))

if not os.path.exists(test_path):
  logging.error("File doesnt exists: {}".format(train_path))
  raise ValueError("File doesnt exists: {}".format(train_path))

if subtask == 'A':
  id2label = {0: "human", 1: "machine"}
  label2id = {"human": 0, "machine": 1}
elif subtask == 'B':
  id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
  label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
else:
  logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
  raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

set_seed(random_seed)

#get data for train/dev/test sets
train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

# downsample training data to train faster
train_df = train_df.groupby("label").sample(n=2000, random_state=random_seed)

# determine avg text length in tokens
print(int(train_df["text"].map(lambda x: len(x.split(" "))).mean()))

# train detector model
fine_tune(train_df, valid_df, f"{model_name}/subtask{subtask}/{random_seed}", id2label, label2id, model_name)

# test detector model
results, predictions = test(test_df, f"{model_name}/subtask{subtask}/{random_seed}/best/", id2label, label2id)

logging.info(results)
predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
predictions_df.to_json(prediction_path, lines=True, orient='records')

print(results)

target_names = [label for idx, label in id2label.items()]
print(classification_report(test_df["label"], predictions, target_names=target_names))

model_path = f"{model_name}/subtask{subtask}/{random_seed}/best/"

# load tokenizer from saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load best model
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id)

df = pd.DataFrame({"text": ["I'm chatGPT a virtual assistant able to generate texts"], "label":[1], "model":["chatGPT"], "source": ["reddit"], "id": 0 })
test_dataset = Dataset.from_pandas(df)

tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# create Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# get logits from predictions and evaluate results using classification report
predictions = trainer.predict(tokenized_test_dataset)
prob_pred = softmax(predictions.predictions, axis=-1)
preds = np.argmax(predictions.predictions, axis=-1)

print("label: %s probability: %s" % (id2label[preds[0]], prob_pred[0][preds[0]]))