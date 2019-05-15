import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Constants = {"<PAD>": 0, "<OOV>": 1, "<SOS>": 2, "<EOS>": 3}


class Beam(object):
    def __init__(self, size):

        self.size = size
        self.done = False

        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(device)
        self.all_scores = []
        self.all_length = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(size).fill_(
            Constants["<PAD>"]).to(device)]
        self.nextYs[0][0] = Constants["<SOS>"]
        self.nextYs_true = [torch.LongTensor(size).fill_(
            Constants["<PAD>"]).to(device)]
        self.nextYs_true[0][0] = Constants["<SOS>"]

        # The attentions (matrix) for each time.
        self.attn = []

        # is copy for each time
        self.isCopy = []

    def getCurrentState(self):
        # Get the outputs for the current timestep.
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        # Get the backpointers for the current timestep.
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, wordLk, copyLk, attnOut):
        numWords = wordLk.size(1)
        numSrc = copyLk.size(1)
        numAll = numWords + numSrc
        allScores = torch.cat((wordLk, copyLk), dim=1)

        if len(self.prevKs) > 0:
            finish_index = self.nextYs[-1].eq(Constants["<EOS>"])
            if any(finish_index):
                allScores.masked_fill_(
                    finish_index.unsqueeze(1).expand_as(allScores),
                    -float('inf'))
                for i in range(self.size):
                    if self.nextYs[-1][i] == Constants["<EOS>"]:
                        allScores[i][Constants["<EOS>"]] = 0
            # set up the current step length
            cur_length = self.all_length[-1]
            for i in range(self.size):
                cur_length[i] += 0 if self.nextYs[-1][i] == Constants["<EOS>"] else 1

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            prev_score = self.all_scores[-1]
            now_acc_score = allScores + prev_score.unsqueeze(1).expand_as(allScores)
            beamLk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
        else:
            self.all_length.append(torch.FloatTensor(self.size).fill_(1).to(device))
            beamLk = allScores[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numAll
        predict = bestScoresId - prevK * numAll
        isCopy = predict.ge(
            torch.LongTensor(self.size).fill_(numWords).to(device)).long()
        final_predict = predict * (1 - isCopy) + isCopy * Constants["<OOV>"]

        if len(self.prevKs) > 0:
            self.all_length.append(cur_length.index_select(0, prevK))
            self.all_scores.append(
                now_acc_score.view(-1).index_select(0, bestScoresId))
        else:
            self.all_scores.append(self.scores)

        self.prevKs.append(prevK)
        self.nextYs.append(final_predict)
        self.nextYs_true.append(predict)
        self.isCopy.append(isCopy)
        self.attn.append(attnOut.index_select(0, prevK))

        # End condition is when every one is EOS.
        if all(self.nextYs[-1].eq(Constants["<EOS>"])):
            self.done = True

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def getBest(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, k):
        hyp, attn = [], []
        isCopy, copyPos = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            attn.append(self.attn[j][k])
            isCopy.append(self.isCopy[j][k])
            copyPos.append(self.nextYs_true[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1], isCopy[::-1], copyPos[::-1], torch.stack(attn[::-1])
