import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import evaluate
from torch.utils.tensorboard import SummaryWriter


################################################
##       Part2 --- Language Modeling          ##
################################################   
class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights).float())
            self.embedding.weight.requires_grad = not freeze_embeddings
        else:  # train from scratch embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # TODO: Define **bi-directional** LSTM layer with `num_layers` and `dropout_rate`.
        ## Hint: what is the input/output dimensions? how to set the bi-directional model?
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0,  # Dropout on all but last layer
                            bidirectional=True,  # Make the LSTM bi-directional
                            batch_first=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        # TODO: Define the feedforward layer with `num_layers` and `dropout_rate`.
        ## Hint: what is the input/output dimensions for a bi-directional LSTM model?
        self.fc = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, input_id):
        embedding = self.dropout(self.embedding(input_id))
        
        # TODO: Get output from (LSTM layer-->dropout layer-->feedforward layer)
        # Embedding layer

        # LSTM layer
        lstm_output, (hidden, cell) = self.lstm(embedding)
        
        # Dropout and feedforward layer
        output = self.dropout(lstm_output)
        output = self.fc(output)
        return output
    
def train_lstm(model, train_loader, optimizer, criterion, device=0, tensorboard_path="./tensorboard"):
    """
    Main training pipeline. Implement the following:
    - pass inputs to the model
    - compute loss
    - perform forward/backward pass.
    """
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    running_loss = 0
    epoch_loss = 0
    for i, data in enumerate(tqdm(train_loader)):
        # get the inputs
        inputs = data.to(device)
        
        # TODO: get the language modelling labels form inputs
        labels = inputs[:, 1:].to(device)

        # TODO: Implement forward pass. Compute predicted y by passing x to the model
        y_pred = model(inputs)

        y_pred = y_pred[:, :-1, :].permute(0, 2, 1)

        # TODO: Compute loss
        loss = criterion(y_pred, labels)
        
        # TODO: Implement Backward pass. 
        # Hint: remember to zero gradients after each update. 
        # You can add several lines of code here.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i>0 and i % 500 == 0. :
            print(f'[Step {i + 1:5d}] loss: {running_loss / 500:.3f}')
            tb_writer.add_scalar("lstm/train/loss", running_loss / 500, i)
            running_loss = 0.0

    tb_writer.flush()
    tb_writer.close()
    print(f'Epoch Loss: {(epoch_loss / len(train_loader)):.4f}')
    return epoch_loss / len(train_loader)

def test_lstm(model, test_loader, criterion, device=0):
    """
    Main testing pipeline. Implement the following:
    - get model predictions
    - compute loss
    - compute perplexity.
    """
 
    # Testing loop
    batch_loss = 0

    model.eval()
    for data in tqdm(test_loader):
        # get the inputs
        inputs = data.to(device)
        labels = data[:, 1:].to(device)

        # TODO: Run forward pass to get model prediction.
        y_pred = model(inputs)
        y_pred = y_pred[:, :-1, :].permute(0, 2, 1)
        
        # TODO: Compute loss
        loss = criterion(y_pred, labels)
        batch_loss += loss.item()

    test_loss = batch_loss / len(test_loader)
    
    # TODO: Get test perplexity using `test_loss``
    perplexity = math.exp(test_loss)    
    print(f'Test loss: {test_loss:.3f}')
    print(f'Test Perplexity: {perplexity:.3f}')
    return test_loss, perplexity

################################################
##       Part3 --- Finetuning          ##
################################################ 

class Encoder(nn.Module):
    def __init__(self, pretrained_encoder, hidden_size):
        super(Encoder, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.hidden_size = hidden_size
        # projection layer
        self.projection = nn.Linear(2*hidden_size, hidden_size)
    def forward(self, input_ids, input_mask):
        # TODO: Implement forward pass.
        # Hint 1: You should take into account the fact that pretrained encoder is bidirectional.
        # Hint 2: Check out the LSTM docs (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
        # Hint 3: Do we need all the components of the pretrained encoder?
        # embed the input
        embeddings = self.pretrained_encoder.embedding(input_ids)
        # pass the embeddings through the LSTM
        encoder_outputs, (h_n , c_n) = self.pretrained_encoder.lstm(embeddings)
        # project the encoder outputs 
        encoder_outputs = self.projection(encoder_outputs)
        encoder_hidden = h_n 
        return encoder_outputs, encoder_hidden

# The initial architecture of this class was generated by ChatGPT. 
# I then proceeded to debug it and made some additions to make 
# the shapes of each operation matches and ensure that the model works.
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.query_weights = nn.Linear(hidden_size, hidden_size)
        self.value_weights = nn.Linear(hidden_size, hidden_size)
        self.combined_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, query, values, mask=None):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code

        # Trasnform weights
        query_transformed = self.query_weights(query)

        # Transform values
        values_transformed = self.value_weights(values)

        # Reshape weights to match values
        query_transformed = query_transformed.repeat(1,values_transformed.shape[1],1)

        # Compute result
        results = query_transformed + values_transformed
        tanh_values = torch.tanh(results)

        # Compute scores
        scores = self.combined_weights(tanh_values).squeeze(-1) 

        # Masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=1) 

        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [batch_size, hidden_size]

        return context, attention_weights

# The initial architecture of this class was generated by ChatGPT. 
# I then proceeded to debug it and made some additions to make 
# the shapes of each operation matches and ensure that the model works.
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, bos_token_id, dropout_rate=0.15, encoder_embedding=None):
        super(Decoder, self).__init__()
        # Note: feel free to change the architecture of the decoder if you like
        if encoder_embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = encoder_embedding
        self.attention = AdditiveAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_size, output_size)
        self.bos_token_id = bos_token_id
        
    def forward(self, encoder_outputs, encoder_hidden, input_mask, target_tensors=None, device=0):
        # TODO: Implement forward pass.
        # Note: this part requires several lines of code
        # Hint: Use target_tensors to handle training and inference appropriately
        
        batch_size = encoder_outputs.size(0)
        input_length = encoder_outputs.size(1)
        output_size = self.embedding.num_embeddings

        # Initialize output tensor to store decoder outputs
        decoder_outputs = torch.zeros(batch_size, input_length, output_size).to(device)
        
        # Prepare the first input to the decoder 
        decoder_input = torch.tensor([self.bos_token_id] * batch_size, dtype=torch.long).to(device)

        # Initialize the hidden state of the decoder
        decoder_hidden = encoder_hidden[-1].unsqueeze(1)
    
        for t in range(1, input_length):
            # Embed the input token
            decoder_input_embedded = self.embedding(decoder_input).unsqueeze(1)  

            # Attention 
            context, _ = self.attention(decoder_hidden, encoder_outputs, input_mask)
    
            # Combine embedded input token and context
            rnn_input = torch.cat((decoder_input_embedded, context.unsqueeze(1)), -1)
    
            # Pass combined input through RNN
            rnn_output, decoder_hidden = self.gru(rnn_input, decoder_hidden.permute(1,0,2))
            decoder_hidden = decoder_hidden.permute(1,0,2)

            # Apply dropout
            output = self.out(rnn_output.squeeze(1))

            # Store the output
            decoder_outputs[:, t, :] = output
    
            # Use teacher forcing or predict the next input
            if target_tensors is not None and t < input_length:
                decoder_input = target_tensors[:, t]  # Next input is current target
            else:
                top1 = output.max(1)[1]
                decoder_input = top1
    
        return decoder_outputs, decoder_hidden

    
class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, bos_token_id, dropout_rate=0.15, pretrained_encoder=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(pretrained_encoder, hidden_size)
        # the embeddings in the encoder and decoder are tied as they're both from the same language
        self.decoder = Decoder(hidden_size, output_vocab_size, bos_token_id, dropout_rate, pretrained_encoder.embedding)

    def forward(self, inputs, input_mask, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs, input_mask)
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets)
        return decoder_outputs, decoder_hidden

def seq2seq_eval(model, eval_loader, criterion, device=0):
    model.eval()
    epoch_loss = 0
    
    for i, data in tqdm(enumerate(eval_loader), total=len(eval_loader)):
        # TODO: Get the inputs
        input_ids, target_ids, input_mask = data['input_ids'].to(device), data['output_ids'].to(device), data['input_mask'].to(device)

        # TODO: Forward pass
        decoder_outputs, decoder_hidden = model(input_ids, input_mask, target_ids)
    
        batch_max_seq_length = target_ids.size(1)
        labels = target_ids

        # TODO: Compute loss

        # Softmax layer
        logsoft = nn.LogSoftmax(dim=2)
        decoder_outputs = logsoft(decoder_outputs)

        loss = criterion(decoder_outputs.permute(0,2,1), labels)
        epoch_loss += loss.item()

    model.train()

    return epoch_loss / len(eval_loader)

def seq2seq_train(model, train_loader, eval_loader, optimizer, criterion, num_epochs=10, device=0, tensorboard_path="./tensorboard"):
    tb_writer = SummaryWriter(tensorboard_path)
    # Training loop
    model.train()
    best_eval_loss = 1e3 # used to do early stopping

    for epoch in tqdm(range(num_epochs), leave=False, position=0):
        running_loss = 0
        epoch_loss = 0
        
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=1):
            # TODO: Get the inputs
            input_ids, target_ids, input_mask = data['input_ids'].to(device), data['output_ids'].to(device), data['input_mask'].to(device)
            

            # Forward pass
            decoder_outputs, decoder_hidden = model(input_ids, input_mask, target_ids)
            
            batch_max_seq_length = train_loader.dataset.max_seq_length
            labels = target_ids

            # TODO: Compute loss
            logsoft = nn.LogSoftmax(dim=2)
            
            # Softmax layer
            decoder_outputs = logsoft(decoder_outputs)
            loss = criterion(decoder_outputs.permute(0,2,1), labels)
            epoch_loss += loss.item()
            
            # TODO: Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            running_loss += loss.item()
            if i % 100 == 99. :    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} | Train Loss: {(epoch_loss / len(train_loader)):.4f}')
        eval_loss = seq2seq_eval(model, eval_loader, criterion, device=device)
        print(f'Epoch {epoch + 1} | Eval Loss: {(eval_loss):.4f}')
        tb_writer.add_scalar("ec-finetune/loss/train", epoch_loss / len(train_loader), epoch)
        tb_writer.add_scalar("ec-finetune/loss/eval", eval_loss, epoch)
        
        # TODO: Perform early stopping based on eval loss
        # Make sure to flush the tensorboard writer and close it before returning

        epochs_without_improvement = 3 
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            counter = 0
        else:
            counter += 1
            if counter > epochs_without_improvement:
                break

    tb_writer.flush()
    tb_writer.close()
    return epoch_loss / len(train_loader)

def seq2seq_generate(model, test_loader, tokenizer, device=0):
    generations = []

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        # TODO: get the inputs
        input_ids, target_ids, input_mask = data['input_ids'].to(device), data['output_ids'].to(device), data['input_mask'].to(device)

        # TODO: Forward pass
        outputs, _ = model(input_ids, input_mask)

        # TODO: Decode outputs to natural language text
        # Note we expect each output to be a string, not list of tokens here
    for o_id, output in enumerate(outputs):
            generations.append({"input": tokenizer.decode(input_ids[o_id].tolist()),
                                "reference": ' '.join(tokenizer.decode(target_ids[o_id].tolist())),
                                "prediction": ' '.join(tokenizer.decode(output.argmax(dim=1).tolist()))})

    
    return generations

def evaluate_rouge(generations):
    # TODO: Implement ROUGE evaluation
    references = [g['reference'] for g in generations]
    predictions = [g['prediction'] for g in generations]

    rouge = evaluate.load('rouge')

    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return rouge_scores

def t5_generate(dataset, model, tokenizer, device=0):
    # TODO: Implement T5 generation
    model.eval()
    generations = []
    model.to(device)

    for sample in tqdm(dataset, total=len(dataset)):
        reference = tokenizer.decode(sample["label_ids"].unsqueeze(0)[0],skip_special_tokens=True)

        # Hint: use huggingface text generation
        input_ids = sample["input_ids"].unsqueeze(0).to(device) 
        outputs = model.generate(input_ids, max_length=128, )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        generations.append({
            "input": input_text, 
            "reference": reference, 
            "prediction": prediction})
    
    return generations