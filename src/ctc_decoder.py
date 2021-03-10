from __future__ import division
from __future__ import print_function
import numpy as np
import torch


class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal = 0 # blank and non-blank
        self.prNonBlank = 0 # non-blank
        self.prBlank = 0 # blank
        self.prText = 1 # LM score
        self.lmApplied = False # flag if LM was already applied to this beam
        self.labeling = () # beam-labeling


class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [x.labeling for x in sortedBeams]


def applyLM(parentBeam, childBeam, classes, lm):
    "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
    if lm and not childBeam.lmApplied:
        c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
        c2 = classes[childBeam.labeling[-1]] # second char
        lmFactor = 0.01 # influence of language model
        bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
        childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
        childBeam.lmApplied = True # only apply LM once per beam entry


def addBeam(beamState, labeling):
    "add beam if it does not yet exist"
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, lm, beamWidth=25, blankIdx=0):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()

        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beamWidth]

        # go over best beams
        for labeling in bestLabelings:

            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
            curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            for c in range(maxC - 1):
                # add new char to current beam-labeling
                newLabeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)
                
                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank
                
                # apply LM
                applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

     # sort by probability
    bestLabeling = last.sort()[0] # get most probable labeling

    # map labels to chars
    res = []#''
    for l in bestLabeling:
        res += classes[l]

    return res


def decode(hypothesis, blank=0, lm=None):
    return greedy_decode(hypothesis, blank=blank)
    #return np.array(ctcBeamSearch(hypothesis, [[i] for i in range(hypothesis.shape[1]+1)], lm=lm)).flatten()


def greedy_decode(hypothesis, blank=0):
    maxs = np.zeros(hypothesis.shape[0])
    for t, P in enumerate(hypothesis):
        maxs [t] = np.argmax(P)
    results = []
    accept = True
    current = -1
    for t in range(hypothesis.shape[0]):
        if maxs[t] != current:
            accept = True
        else:
            accept = False
        if accept:# and maxs[t] != blank:
            results.append(maxs[t])
        current = maxs[t]
    return torch.from_numpy(np.array(results))


def CTC_confidence(L_CTC):
    """
    convidence score of a prediction based on the CTC Loss
    :param L_CTC: CTC loss values
    :return: convidence scores based on loss values
    """
    confidence = 1 / torch.exp(L_CTC)
    return confidence


def torch_confidence(log_P, dset, blank=0):
    """
    calculates the confidence based on marginalization, marginalization is not carried out directly but rather by using
    PyTorch's CTCLoss.
    Loss values are being handed to CTC_convidence
    """
    def tens_convert(x):
        if not isinstance(x, torch.Tensor):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            else:
                x = torch.tensor(x)
        x = x.type(torch.float32)
        return x
    log_P = tens_convert(log_P)
    targets = []
    len_targets = []
    for i in range(log_P.shape[1]):
        tgt = decode(log_P[:,i,:])
        tgt = dset.embedding_to_word(tgt)
        tgt = dset.word_to_embedding(tgt)
        targets.append(tgt)
        len_targets.append(len(tgt))
    len_in = [log_P.shape[0] for _ in range(log_P.shape[1])]
    targets = torch.cat(targets)
    L_CTC = torch.nn.functional.ctc_loss(log_P, targets, len_in, len_targets, blank=blank)
    conf = CTC_confidence(L_CTC)
    return targets, conf

