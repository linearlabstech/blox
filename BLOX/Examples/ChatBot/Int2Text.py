


from .GreedyDecoder import GreedyDecoder
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
class Evaluator(nn.Module):

    def __init__(self,efname,dfname,vfname)
        self.searcher = GreedyDecoder(efname,dfanme)
        self.voc = Voc(vfname)

    def forward(self,x,max_length=10):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, x)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words