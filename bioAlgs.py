import re
import random
import numpy as np


AA2MASS = {'G' : 57, 'A' : 71, 'S' : 87, 'P' : 97, 'V' : 99, 'T' : 101, 'C' : 103, 'I' : 113, 'L' : 113, 'N' : 114, 'D' : 115, 'K' : 128, 'Q' : 128, 'E' : 129, 'M' : 131, 'H' : 137, 'F' : 147, 'R' : 156, 'Y' : 163, 'W' : 186}


def patternCount(genome, pattern):
    count = 0
    for i in range(len(genome)-len(pattern)+1):
        if genome[i:i+len(pattern)] == pattern:
            count += 1
    return count


def frequencyTable(genome, k):
    freqMap = dict()
    for i in range(len(genome)-k):
        freqMap[genome[i: i + k]] = freqMap.get(genome[i: i + k], 0) + 1
    return freqMap


def frequentWords(genome, k):
    frequentPatterns = []
    freqMap = frequencyTable(genome, k)
    for pattern, freq in freqMap.items():
        if freq == max(freqMap.values()):
            frequentPatterns.append(pattern)
    return frequentPatterns


def reverseComplement(pattern):
    complement = ''
    for n in pattern:
        if n == 'A':
            complement = 'T' + complement
        elif n == 'T':
            complement = 'A' + complement
        elif n == 'C':
            complement = 'G' + complement
        elif n == 'G':
            complement = 'C' + complement
    return complement


def patternMatching(pattern, genome):
    positions = []
    for i in range(len(genome)-len(pattern)):
        if genome[i:i+len(pattern)] == pattern:
            positions.append(i)
    return positions


def findClumps(genome, k, L, t):
    patterns = []
    for i in range(len(genome)-L):
        freqMap = frequencyTable(genome[i:i+L], k)
        for pattern, freq in freqMap.items():
            if freq >= t and pattern not in patterns:
                patterns.append(pattern)
    return patterns


def skew(genome):
    curve = [0]
    min_val = 0
    for n in genome:
        if n == 'C':
            curve.append(curve[-1] - 1)
        elif n == 'G':
            curve.append(curve[-1] + 1)
        else:
            curve.append(curve[-1])
        if curve[-1] < min_val:
            min_val = curve[-1]
    mins = []
    for i in range(len(curve)):
        if curve[i] == min_val:
            mins.append(i)
    return mins


def hammingDistance(pattern1, pattern2):
    return sum(n1 != n2 for n1, n2 in zip(pattern1, pattern2))


def approximatePatternMatching(pattern, genome, mismatches):
    positions = []
    for i in range(len(genome)-len(pattern)):
        if hammingDistance(genome[i:i+len(pattern)], pattern) <= mismatches:
            positions.append(i)
    return positions


def neighbors(pattern, mismatches):
    nucleotides = set(['A', 'T', 'G', 'C'])
    if mismatches == 0:
        return [pattern]
    if len(pattern) == 1:
        return list(nucleotides)
    neighbors_list = neighbors(pattern[1:], mismatches - 1)
    neighborhood = [n + neighbor for neighbor in neighbors_list for n in nucleotides - set([pattern[0]])]
    if mismatches < len(pattern):
        neighbors_list = neighbors(pattern[1:], mismatches)
        neighborhood += [pattern[0] + neighbor for neighbor in neighbors_list]
    return neighborhood


def frequentWordsWithMismatches(genome, k, d):
    frequentPatterns = []
    freqMap = {}
    for i in range(len(genome) - k):
        neighborhood = neighbors(genome[i:i+k], d)
        for neighbor in neighborhood:
            freqMap[neighbor] = freqMap.get(neighbor, 0) + 1
    for pattern, freq in freqMap.items():
        if freq == max(freqMap.values()):
            frequentPatterns.append(pattern)
    return frequentPatterns


def frequentWordsWithMismatchesAndReverseComplements(genome, k, d):
    frequentPatterns = []
    freqMap = {}
    for i in range(len(genome) - k):
        neighborhood = neighbors(genome[i:i+k], d)
        for neighbor in neighborhood:
            freqMap[neighbor] = freqMap.get(neighbor, 0) + 1
            freqMap[reverseComplement(neighbor)] = freqMap.get(reverseComplement(neighbor), 0) + 1
    for pattern, freq in freqMap.items():
        if freq == max(freqMap.values()):
            frequentPatterns.append(pattern)
    return frequentPatterns


def motifEnumeration(dna, k, d):
    patterns = set()
    for genome1 in dna:
        for i in range(len(genome1) - k + 1):
            neighborhood = neighbors(genome1[i:i+k], d)
            for neighbor in neighborhood:
                for genome2 in dna:
                    isMotif = False
                    for j in range(len(genome2) - k + 1):
                        if hammingDistance(neighbor, genome2[j:j+k]) <= d:
                            isMotif = True
                    if not isMotif:
                        break
                if isMotif:
                    patterns.add(neighbor)
    return list(patterns)


def mostProbableKmer(genome, k, profile):
    nucleotides = ['A', 'C', 'G', 'T']
    alp = {nucleotides[i] : i for i in range(4)}
    kmer = genome[:k]
    prob = 0
    for i in range(len(genome) - k + 1):
        cur_prob = 1
        for j in range(k):
            cur_prob *= profile[alp[genome[i:i+k][j]]][j]
        if cur_prob > prob:
            prob = cur_prob
            kmer = genome[i:i+k]
    return kmer


def motifsProfile(motifs, laplace=False):
    profile = [[],[],[],[]]
    for i in range(len(motifs[0])):
        if laplace:
            alp = {n : 1 for n in ['A', 'C', 'G', 'T']}
        else:
            alp = {n : 0 for n in ['A', 'C', 'G', 'T']}
        for motif in motifs:
            alp[motif[i]] += 1
        profile[0].append(alp['A'] / sum(alp.values()))
        profile[1].append(alp['C'] / sum(alp.values()))
        profile[2].append(alp['G'] / sum(alp.values()))
        profile[3].append(alp['T'] / sum(alp.values()))
    return profile


def motifScore(motifs):
    score = 0
    for i in range(len(motifs[0])):
        alp = {n : 0 for n in ['A', 'C', 'G', 'T']}
        for motif in motifs:
            alp[motif[i]] += 1
        score += (len(motifs) - max(alp.values()))
    return score


def greedyMotifSearch(dna, k, t, laplace=False):
    bestMotifs = [row[:k] for row in dna]
    for j in range(len(dna[0]) - k + 1):
        motifs = [dna[0][j:j+k]]
        for i in range(1, t):
            profile = motifsProfile(motifs, laplace)
            motifs.append(mostProbableKmer(dna[i], k, profile))
        if motifScore(motifs) < motifScore(bestMotifs):
            bestMotifs = motifs
    return bestMotifs


def randomizedMotifSearch(dna, k, t):
    motifs = []
    for row in dna:
        start = random.randint(0, len(row) - k)
        motifs.append(row[start:start+k])
    bestMotifs = motifs
    while True:
        profile = motifsProfile(motifs, laplace=True)
        motifs = [mostProbableKmer(row, k, profile) for row in dna]
        if motifScore(motifs) < motifScore(bestMotifs):
            bestMotifs = motifs
        else:
            return bestMotifs


def probability(kmer, profile):
    nucleotides = ['A', 'C', 'G', 'T']
    alp = {nucleotides[i] : i for i in range(4)}
    prob = 1
    for i in range(len(kmer)):
        prob *= profile[alp[kmer[i]]][i]
    return prob


def profileRandomKmer(genome, k, profile):
    probs = [probability(genome[i:i+k], profile) for i in range(len(genome) - k + 1)]
    probs = [prob / sum(probs) for prob in probs]
    i = random.choices(range(len(probs)), weights=probs)[0]
    return genome[i:i+k]


def gibbsSampler(dna, k, t, n):
    motifs = []
    for row in dna:
        start = random.randint(0, len(row) - k)
        motifs.append(row[start:start+k])
    bestMotifs = motifs
    for _ in range(n):
        i = random.randint(0, t - 1)
        motifs.pop(i)
        profile = motifsProfile(motifs, laplace=True)
        kmer_i = profileRandomKmer(dna[i], k, profile)
        motifs.insert(i, kmer_i)
        if motifScore(motifs) < motifScore(bestMotifs):
            bestMotifs = motifs
    return bestMotifs


def distanceBetweenPatternAndStrings(pattern, dna):
    distance = 0
    for row in dna:
        hamDist = 1e+9
        for i in range(len(row) - len(pattern) + 1):
            if hamDist > hammingDistance(pattern, row[i:i+len(pattern)]):
                hamDist = hammingDistance(pattern, row[i:i+len(pattern)])
        distance += hamDist
    return distance


def pathToGenome(path):
    return path[0] + ''.join([p[-1] for p in path[1:]])


def composition(genome, k):
    return [genome[i:i+k] for i in range(len(genome) - k + 1)]


def deBruijn(patterns):
    dB = {}
    for pattern in patterns:
        dB[pattern[:-1]] = dB.get(pattern[:-1], []) + [pattern[1:]]
    return dB


def eulerianCycle(graph, start_v=None):
    start_v = random.choice(list(graph.keys())) if start_v is None else start_v
    cycle = [start_v]
    while start_v in graph and len(graph[start_v]) != 0:
        rand_next_v = random.choice(list(range(len(graph[start_v]))))
        next_v = graph[start_v][rand_next_v]
        cycle.append(next_v)
        if len(graph[start_v]) == 1:
            del graph[start_v]
        else:
            del graph[start_v][rand_next_v]
        start_v = next_v
    while len(graph):
        for cur_v in cycle:
            if cur_v in graph:
                i = cycle.index(cur_v)
                start_v = cur_v
                cur_cycle = [start_v]
                while len(graph[start_v]):
                    rand_next_v = random.choice(list(range(len(graph[start_v]))))
                    next_v = graph[start_v][rand_next_v]
                    cur_cycle.append(next_v)
                    if len(graph[start_v]) == 1:
                        del graph[start_v]
                    else:
                        del graph[start_v][rand_next_v]
                    start_v = next_v
                    if start_v not in graph:
                        cycle = cycle[:i] + cur_cycle + cycle[i+1:]
                        break
    return cycle


def eulerianPath(graph):
    edges = {}
    for v_from, v_to in graph.items():
        edges[v_from] = (len(v_to), edges[v_from][1]) if v_from in edges != 2 else (len(v_to), 0)
        for v in v_to:
            edges[v] = (edges[v][0], edges[v][1] + 1)  if v in edges != 2 else (0, 1)
    for v, edge in edges.items():
        if edge[0] > edge[1]:
            return eulerianCycle(graph, v)
    return eulerianCycle(graph, list(edges.keys())[0])


def stringReconstruction(patterns):
    dB = deBruijn(patterns)
    path = eulerianPath(dB)
    return pathToGenome(path)


def kUniversalCircularString(k):
    patterns = []
    i = 0
    while len('{0:b}'.format(i)) <= k:
        patterns.append('{0:b}'.format(i))
        i += 1
    patterns = ['0' * (k - len(p)) + p for p in patterns]
    dB = deBruijn(patterns)
    cycle = eulerianCycle(dB, patterns[0][1:])
    return ''.join([path[-1] for path in cycle[:-1]])


def pairedReads(paires):
    return [kmers.split('|') for kmers in paires]


def pairDeBruijn(patterns):
    dB = {}
    for pattern in patterns:
        dB[(pattern[0][:-1],pattern[1][:-1])] = dB.get((pattern[0][:-1],pattern[1][:-1]), []) + [(pattern[0][1:],pattern[1][1:])]
    return dB


def stringReconstructionFromReadPairs(k, d, kmersPaires):
    dB = pairDeBruijn(kmersPaires)
    path = eulerianPath(dB)
    return gappedGenomePathString(k, d, path)


def gappedGenomePathString(k, d, kmersPaires):
    first_kmers = ''.join([kmer[0][0] for kmer in kmersPaires[:-1]]) + kmersPaires[-1][0]
    second_kmers = ''.join([kmer[1][0] for kmer in kmersPaires[:-1]]) + kmersPaires[-1][1]
    return first_kmers + second_kmers[len(second_kmers)-k-d:]


def contigGeneration(patterns):
    dB = deBruijn(patterns)
    reverse_dB={}
    for k, v in dB.items():
        for i in v:
            reverse_dB[i] = reverse_dB.get(i, []) + [k]
    b1 = [v for v in set(dB.keys()).union(reverse_dB.keys()) if v not in dB or len(dB[v]) != 1]
    b2 = [v for v in set(dB.keys()).union(reverse_dB.keys()) if v not in reverse_dB or len(reverse_dB[v]) != 1]
    contigs = []
    for from_v in set(b1).union(b2):
        if from_v in dB:
            for to_v in dB[from_v]:
                contig = [from_v, to_v]
                while not to_v in set(b1).union(b2):
                    to_v = dB[to_v][0]
                    contig.append(to_v)
                contigs.append(contig)
    return [contig[0] + ''.join([c[-1] for c in contig[1:]]) for contig in contigs]


def DNA2RNA(dna):
    return dna.replace('T', 'U')


def RNA2AA(rna):
    codon2aa = {'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
                'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', 'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
                'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*', 'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W', 'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
                'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    }
    aa = ''
    for i in range(0, len(rna) - 2, 3):
        if codon2aa[rna[i:i+3]] == '*':
            break
        aa += codon2aa[rna[i:i+3]]
    return aa


def peptideEncoding(genome, peptide):
    substrings = []
    for i in range(0, len(genome) - 3 * len(peptide) + 1):
        pattern = genome[i:i+3*len(peptide)]
        complement = reverseComplement(pattern)
        if RNA2AA(DNA2RNA(pattern)) == peptide or RNA2AA(DNA2RNA(complement)) == peptide:
            substrings.append(pattern)
    return substrings


def countPeptides(mass):
    peptides = [0] * (mass + 1)
    for i in range(mass + 1):
        for k in np.unique(np.array(list(AA2MASS.values()))):
            if i - k == 0:
                peptides[i] += 1
            elif i - k > 0:
                peptides[i] += peptides[i - k]
    return peptides[-1]


def expand(peptides):
    return [peptide + [mass] for peptide in peptides for mass in np.unique(np.array(list(AA2MASS.values())))]


def linearSpectrum(peptide):
    prefixMass, linSpectrum = [0], [0]
    for i in range(len(peptide)):
        prefixMass.append(prefixMass[i] + int(peptide[i]))
    for i in range(len(peptide)):
        for j in range(i+1, len(peptide)+1):
            linSpectrum.append(prefixMass[j] - prefixMass[i])
    return sorted(linSpectrum)


def cyclospectrum(peptide):
    spectrum = [0, sum(peptide)] + [sum((peptide * 2)[j:j + i]) for i in range(1, len(peptide)) for j in range(len(peptide))]
    return sorted(spectrum)


def isConsistent(peptide, spectrum):
    if sum(peptide) > spectrum[-1] - list(AA2MASS.values())[0]:
        return False
    for mass in linearSpectrum(peptide):
        if not mass in spectrum:
            return False
    return True


def cyclopeptideSequencing(spectrum):
    peptides = [[]]
    output = []
    while len(peptides) != 0:
        peptides = expand(peptides)
        for peptide in peptides:
            if sum(peptide) == spectrum[-1]:
                if cyclospectrum(peptide) == spectrum:
                    output.append(peptide)
                peptides = [p for p in peptides if p != peptide]
            elif not isConsistent(peptide, spectrum):
                peptides = [p for p in peptides if p != peptide]
    return output


def score(pepSpectrum, spectrum):
    score = 0
    for i in spectrum:
        if i in pepSpectrum:
            pepSpectrum = [j for j in pepSpectrum if j != i]
            score += 1
    return score


def cut(peptides, spectrum, n):
    scores = [[score(linearSpectrum(peptide), spectrum), peptide] for peptide in peptides]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scores[:n]]


def leaderboardCyclopeptideSequencing(spectrum, n):
    leaderboard = [[]]
    leaderPeptide = []
    while len(leaderboard) != 0:
        leaderboard = expand(leaderboard)
        for peptide in leaderboard:
            if sum(peptide) == spectrum[-1]:
                if score(cyclospectrum(peptide), spectrum) > score(cyclospectrum(leaderPeptide), spectrum):
                    leaderPeptide = peptide.copy()
            elif sum(peptide) > spectrum[-1]:
                leaderboard = [p for p in leaderboard if p != peptide]
        leaderboard = cut(leaderboard, spectrum, n)
    return leaderPeptide


def convExpand(peptides, spectrum, m):
    conv = [m2 - m1 for m1 in spectrum for m2 in spectrum if m2 > m1]
    convRest = [i for i in conv if 56 < i < 201]
    convCnt= {}
    for i in convRest:
        convCnt[i] = convCnt.get(i, 0) + 1
    masses = np.unique(np.array([i for i in convRest if convCnt[i] >= sorted(list(convCnt.values()), reverse=True)[m]]))
    return [peptide + [mass] for peptide in peptides for mass in sorted(masses)]


def convolutionCyclopeptideSequencing(spectrum, m, n):
    leaderboard = [[]]
    leaderPeptide = []
    while len(leaderboard) != 0:
        leaderboard = convExpand(leaderboard, spectrum, m)
        for peptide in leaderboard:
            if sum(peptide) == spectrum[-1]:
                if score(cyclospectrum(peptide), spectrum) > score(cyclospectrum(leaderPeptide), spectrum):
                    leaderPeptide = peptide.copy()
            elif sum(peptide) > spectrum[-1]:
                leaderboard = [p for p in leaderboard if p != peptide]
        leaderboard = cut(leaderboard, spectrum, n)
    return leaderPeptide


def recurseChange(money, coins, cntrs):
    minNumberOfCoins = money
    if money in cntrs: return cntrs[money], coins, cntrs
    for coin in coins:
        if coin <= money:
            numberOfCoins, coins, cntrs = recurseChange(money - coin, coins, cntrs)
            minNumberOfCoins = numberOfCoins + 1 if numberOfCoins + 1 < minNumberOfCoins else minNumberOfCoins
    cntrs[money] = minNumberOfCoins
    return minNumberOfCoins, coins, cntrs


def changeProblem(money, coins):
    import sys
    sys.setrecursionlimit(money)
    minNumberOfCoins, _, _ =  recurseChange(money, coins, {coin : 1 for coin in coins})
    return minNumberOfCoins


def manhattanTourist(n, m, down, right):
    matrix = [[0 for x in range(m + 1)] for y in range(n + 1)]
    matrix[0][0] = 0
    for i in range(1, n + 1):
        matrix[i][0] = matrix[i - 1][0] + down[i - 1][0]
    for j in range(1, m + 1):
        matrix[0][j] = matrix[0][j - 1] + right[0][j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            matrix[i][j] = max(matrix[i - 1][j] + down[i - 1][j], matrix[i][j - 1] + right[i][j - 1])
    return matrix[n][m]


def readScoreMatrix(path='./BLOSUM62'):
    scoreMatrix = dict()
    with open(path) as f:
        proteins = f.readline().strip().split()
        for row in f:
            row = row.strip().split()
            for i in range(len(proteins)):
                scoreMatrix[(row[0],proteins[i])] = int(row[i + 1])
    return scoreMatrix


def outputLCS(backtrack, aaStr1, aaStr2, i, j):
    str1, str2 = '', ''
    while i and j:
        if backtrack[i][j] == 'diag':
            str1 += aaStr1[i-1]
            str2 += aaStr2[j-1]
            i -= 1
            j -= 1
        elif backtrack[i][j] == 'down':
            str1 += aaStr1[i-1]
            str2 += '-'
            i -= 1
        else:
            str1 += '-'
            str2 += aaStr2[j-1]
            j -= 1
    while i:
        str1 += aaStr1[i-1]
        str2 += '-'
        i -= 1
    while j:
        str1 += '-'
        str2 += aaStr2[j-1]
        j -= 1
    return str1[::-1], str2[::-1]


def LCSBackTrack(aaStr1, aaStr2, scoreMatrix, sigma=0):
    backtrack = [[0 for i in range(len(aaStr2) + 1)] for j in range(len(aaStr1) + 1)]
    alignmentScore = [[0 for i in range(len(aaStr2) + 1)] for j in range(len(aaStr1) + 1)]
    for i in range(1, len(aaStr1) + 1):
        alignmentScore[i][0] = alignmentScore[i-1][0] - sigma
    for j in range(1, len(aaStr2) + 1):
        alignmentScore[0][j] = alignmentScore[0][j-1] - sigma
    for i in range(1, len(aaStr1) + 1):
        for j in range(1,len(aaStr2) + 1):
            alignmentScore[i][j] = max(alignmentScore[i - 1][j] - sigma, alignmentScore[i][j - 1] - sigma,
                                       alignmentScore[i - 1][j - 1] + scoreMatrix[(aaStr1[i - 1], aaStr2[j - 1])])
            if alignmentScore[i][j] == alignmentScore[i - 1][j] - sigma:
                backtrack[i][j] = 'down'
            elif alignmentScore[i][j] == alignmentScore[i][j-1] - sigma:
                backtrack[i][j] = 'right'
            elif alignmentScore[i][j] == alignmentScore[i - 1][j - 1] + scoreMatrix[(aaStr1[i - 1],aaStr2[j - 1])]:
                backtrack[i][j] = 'diag'
    return alignmentScore[len(aaStr1)][len(aaStr2)], backtrack


def globalAlignment(aaStr1, aaStr2, sigma=5):
    scoreMatrix = readScoreMatrix(path='./BLOSUM62')
    alignmentScore, backtrack = LCSBackTrack(aaStr1, aaStr2, scoreMatrix, sigma=5)
    return alignmentScore, *outputLCS(backtrack, aaStr1, aaStr2, len(aaStr1), len(aaStr2))


def outputLocalLCS(backtrack, aaStr1, aaStr2, i, j):
    str1, str2 = aaStr1[:i], aaStr2[:j]
    while i and j:
        if backtrack[i][j] == 0:
            str2 = str2[:j] + '-' + str2[j:]
            i -= 1
        elif backtrack[i][j] == 1:
            str1 = str1[:i] + '-' + str1[i:]
            j -= 1
        elif backtrack[i][j] == 2:
            i -= 1
            j -= 1
        else:
            break
    return str1[i:], str2[j:]


def localLCSBackTrack(aaStr1, aaStr2, scoreMatrix, sigma=0):
    backtrack = [[0 for i in range(len(aaStr2) + 1)] for j in range(len(aaStr1) + 1)]
    alignmentScore = [[0 for i in range(len(aaStr2) + 1)] for j in range(len(aaStr1) + 1)]
    for i in range(1, len(aaStr1) + 1):
        for j in range(1,len(aaStr2) + 1):
            scores = np.array([alignmentScore[i - 1][j] - sigma, alignmentScore[i][j - 1] - sigma,
                               alignmentScore[i - 1][j - 1] + scoreMatrix[(aaStr1[i - 1], aaStr2[j - 1])], 0])
            alignmentScore[i][j] = np.max(scores)
            backtrack[i][j] = np.argmax(scores)
    return alignmentScore, backtrack


def localAlignment(aaStr1, aaStr2, sigma=5):
    scoreMatrix = readScoreMatrix(path='./PAM250')
    alignmentScore, backtrack = localLCSBackTrack(aaStr1, aaStr2, scoreMatrix, sigma=5)
    i, j = np.unravel_index(np.argmax(np.array(alignmentScore)), np.array(alignmentScore).shape)
    return alignmentScore[i][j], *outputLocalLCS(backtrack, aaStr1, aaStr2, i, j)


def outputMultipleLCS(backtrack, aaStr1, aaStr2, aaStr3, i, j, k):
    while i and j and k:
        if backtrack[i][j][k] == 0:
            aaStr2 = aaStr2[:j] + "-" + aaStr2[j:]
            aaStr3 = aaStr3[:k] + "-" + aaStr3[k:]
            i -= 1
        elif backtrack[i][j][k] == 1:
            aaStr1 = aaStr1[:i] + "-" + aaStr1[i:]
            aaStr3 = aaStr3[:k] + "-" + aaStr3[k:]
            j -= 1
        elif backtrack[i][j][k] == 2:
            aaStr1 = aaStr1[:i] + "-" + aaStr1[i:]
            aaStr2 = aaStr2[:j] + "-" + aaStr2[j:]
            k -= 1
        elif backtrack[i][j][k] == 3:
            aaStr2 = aaStr2[:j] + "-" + aaStr2[j:]
            i -= 1
            k -= 1
        elif backtrack[i][j][k] == 4:
            aaStr1 = aaStr1[:i] + "-" + aaStr1[i:]
            j -= 1
            k -= 1
        elif backtrack[i][j][k] == 5:
            aaStr3 = aaStr3[:k] + "-" + aaStr3[k:]
            i -= 1
            j -= 1
        else:
            i -= 1
            j -= 1
            k -= 1
    for r in range(i, max(i, j, k)):
        aaStr1 = aaStr1[:0] + "-" + aaStr1[0:]
    for r in range(j, max(i, j, k)):
        aaStr2 = aaStr2[:0] + "-" + aaStr2[0:]
    for r in range(k, max(i, j, k)):
        aaStr3 = aaStr3[:0] + "-" + aaStr3[0:]
    return aaStr1, aaStr2, aaStr3


def multipleLCSBackTrack(aaStr1, aaStr2, aaStr3, sigma=0):
    backtrack = np.zeros((len(aaStr1) + 1, len(aaStr2) + 1, len(aaStr3) + 1))
    alignmentScore = np.zeros((len(aaStr1) + 1, len(aaStr2) + 1, len(aaStr3) + 1))
    for i in range(1, len(aaStr1) + 1):
        for j in range(1, len(aaStr2) + 1):
            for k in range(1, len(aaStr3) + 1):
                scores = np.array([alignmentScore[i - 1][j][k] - sigma, alignmentScore[i][j - 1][k] - sigma,
                                   alignmentScore[i][j][k - 1] - sigma, alignmentScore[i - 1][j][k - 1],
                                   alignmentScore[i][j - 1][k - 1] - sigma, alignmentScore[i - 1][j - 1][k],
                                   alignmentScore[i-1][j-1][k-1] + 1 if aaStr1[i - 1] == aaStr2[j - 1] == aaStr3[k - 1] else 0])
                alignmentScore[i][j][k] = np.max(scores)
                backtrack[i][j][k] = np.argmax(scores)
    return int(alignmentScore[-1][-1][-1]), backtrack


def multipleLongestCommonSubsequence(aaStr1, aaStr2, aaStr3, sigma=0):
    alignmentScore, backtrack = multipleLCSBackTrack(aaStr1, aaStr2, aaStr3, sigma)
    return alignmentScore, *outputMultipleLCS(backtrack, aaStr1, aaStr2, aaStr3, len(aaStr1), len(aaStr2), len(aaStr3))


def greedySorting(permutation):
    sortPermutation = []
    for k in range(len(permutation)):
        if abs(permutation[k]) != k + 1:
            permByRev = [-permutation[k]]
            for i in range(k+1, len(permutation)):
                permByRev.append(-permutation[i])
                if abs(permutation[i]) == k + 1:
                    permByRev = list(reversed(permByRev))
                    permutation[k:i+1] = permByRev
                    sortPermutation.append(permutation.copy())
        if permutation[k] < 0:
            permutation[k] = -permutation[k]
            sortPermutation.append(permutation.copy())
    return sortPermutation


def numberOfBreaks(permutation):
    permutation = [0] + permutation + [len(permutation) + 1]
    return sum([1 for i in range(len(permutation) - 1) if permutation[i] + 1 != permutation[i + 1]])


def twoBreakDistance(P, Q):
    edges = {}
    for permutation in P + Q:
        for i in range(len(permutation)):
            edges[permutation[i]] = edges.get(permutation[i], []) + [-1 * permutation[(i + 1) % len(permutation)]]
            edges[-1 * permutation[(i + 1) % len(permutation)]] = edges.get(-1 * permutation[(i + 1) % len(permutation)], []) + [permutation[i]]
    cycleCnt = 0
    while len(edges) > 0:
        cur_edge = list(edges.keys())[0]
        while cur_edge in edges:
            node = edges[cur_edge][0]
            if len(edges[cur_edge]) == 1:
                del edges[cur_edge]
            else:
                edges[cur_edge] = edges[cur_edge][1:]
            if edges[node] == [cur_edge]:
                del edges[node]
            else:
                edges[node].remove(cur_edge)
            cur_edge = node
        cycleCnt += 1
    return sum(map(len,P)) - cycleCnt


def chromosomeToCycle(chromosome):
    nodes = []
    for j in range(len(chromosome)):
        i = chromosome[j]
        if i > 0:
            nodes.append(2 * i - 1)
            nodes.append(2 * i)
        else:
            nodes.append(-2 * i)
            nodes.append(-2 * i - 1)
    return nodes


def cycleToChromosome(nodes):
    chromosome = []
    for j in range(len(nodes) // 2):
        if nodes[2 * j] < nodes[2 * j + 1]:
            chromosome.append(nodes[2 * j + 1] // 2)
        else:
            chromosome.append(-nodes[2 * j] // 2)
    return chromosome


def coloredEdges(P):
    edges = set()
    for chromosome in P:
        nodes = chromosomeToCycle(chromosome)
        for j in range(len(nodes) // 2):
            edges.add((nodes[2 * j + 1], nodes[(2 * j + 2) % len(nodes)]))
    return sorted(list(map(list, edges)))


def graphToGenome(genomeGraph):
    genome, used = [], []
    graph = [0] * 2 * len(genomeGraph)
    for edge in genomeGraph:
        graph[edge[0]-1] = edge[1] - 1
        graph[edge[1]-1] = edge[0] - 1
    for edge in genomeGraph:
        start = edge[0]
        if start in used: continue
        used.append(start)
        end = start + 1 if start % 2 else start - 1
        cur = []
        while True:
            cur.append(-(start + 1) // 2 if start % 2 else start // 2)
            used.append(graph[start-1] + 1)
            if graph[start-1] + 1 == end:
                genome.append(cur)
                break
            start = graph[start-1] + 2 if (graph[start-1] + 1) % 2 else graph[start-1]
            used.append(start)
    return genome


def twoBreakOnGenomeGraph(genomeGraph, i, i_, j, j_):
    e1 = [i, i_] if [i, i_] in genomeGraph else [i_, i]
    e2 = [j, j_] if [j, j_] in genomeGraph else [j_, j]
    genomeGraph.remove(e1)
    genomeGraph.remove(e2)
    return genomeGraph + [[i, j], [i_, j_]]


def twoBreakOnGenome(P, i, i_, j, j_):
    genomeGraph = coloredEdges(P)
    genomeGraph = twoBreakOnGenomeGraph(genomeGraph, i, i_, j, j_)
    return graphToGenome(genomeGraph)


def buildGraph(P, Q):
    graph = [[0, 0] for _ in range(len(P) + len(Q))]
    for p, q in zip(P, Q):
        graph[p[0]-1][0] = p[1] - 1
        graph[p[1]-1][0] = p[0] - 1
        graph[q[0]-1][1] = q[1] - 1
        graph[q[1]-1][1] = q[0] - 1
    return graph


def coloredCycles(P, Q):
    coloredCyc, used = [], [0] * (len(P) + len(Q))
    graph = buildGraph(P, Q)
    for v in range(len(P) + len(Q)):
        if used[v]:
            continue
        used[v] = 1
        start, color = v, 0
        cyc = [start + 1]
        while True:
            if graph[v][color] == start:
                coloredCyc.append(cyc)
                break
            v, color = graph[v][color], ~color
            used[v] = 1
            cyc.append(v + 1)
    return coloredCyc


def twoBreakSorting(P, Q):
    sequences = [P]
    while twoBreakDistance(P, Q) > 0:
        for cycle in coloredCycles(coloredEdges(P), coloredEdges(Q)):
            if len(cycle) > 3:
                P = twoBreakOnGenome(P, cycle[0], cycle[1], cycle[3], cycle[2])
                sequences.append(P)
                break
    return sequences


if __name__=='__main__':
    # genome = 'GTCTTTAGTCTTTAGTCTCTTTAGCAATCTTTAGATCTTTAGGATTCTATCTTTAGTCTTTAGGCTGCCGTTCTTTAGGGCATCTTTAGATTCTTTAGTCTTTAGTCTTTAGTGCTGTTTCTTTAGTCTTTAGTCTTTAGTCTTTAGGTCTTTAGTCTTTAGACCAATTCTTTAGCTCTTTAGAAGGAAGGATCTTTAGTCTTTAGAAACGTCTTTAGTCTTTAGCTCTTTAGTCTTTAGTAAGAGTCTTTAGTTTTTACCCGTTCTTTAGGATGATCTTTAGTGATCTTTAGGCTCTTTAGTCTTTAGGGATCTTTAGTCTTTAGCCGAAAAGTTGTCTTTAGTAATGATCTTTAGAGGTCTTTAGGTCTTTAGTCTTTAGCTCTTTAGGGACGGAATCTTTAGCTCTTTAGCGTCCTCTTTAGTCCTCTTTAGTGCTCGACGATACTGTCTTTAGTCTTTAGTAATCTCTTTAGAGTTCTTTAGTCCGTCTTTAGCCTTTATCTTTAGTGATATTTTTGTCTTTAGCGTCTTTAGACATCTCTTTAGTGTCTTTAGGTCTTTAGCGCATCTTTAGTCTTTAGTCTGTCTTTAGTCTTTAGATCTTTAGAACTCTTTAGGGCTAGTCTTTAGCTCTTTAGCAGGTCTTTAGTCTTTAGGGTTTCTTTAGTCGGGCGGTCTTTAGTTCTTTAGTCTTTAGGGCATCTTTAGTCTTTAGTCTTTAGTTCTATCTTTAGTTTCTCTTTAGTCTTTAGTCTTTAGTCTTTAGTCTTTAGATCTTTAGTTCTTTAGATCTTTAGCTCTTTAGGATCCATTCTTTAGTCTTTAGTTCTTTAGCTCTTTAGCATTATCTTTAGTCTTTAGTCTTTAGTCTTTAGGGGATCTTTAGTCTTTAGGAGTCTTTAGGTCTCTTTAGGCGGATATCTTTAG'
    # pattern = 'TCTTTAGTC'
    # ans = patternCount(genome, pattern)


    # genome = 'TCACCGTCTACCACCTCCTCACCGTCTATCACCGTCTATATCGCTGCCACCTCCTCACCGTCTATCACCGTCTATCACCGTCTATATCGCTGTATCGCTGCCACCTCCTATCGCTGTGGGTTTATCACCGTCTATCACCGTCTATATCGCTGAACTTAGAGTCACCGTCTATGGGTTTATATCGCTGCCACCTCCTATCGCTGTGGGTTTAAACTTAGAGTGGGTTTACCACCTCCTGGGTTTATCACCGTCTATGGGTTTAAACTTAGAGTATCGCTGTCACCGTCTATATCGCTGTGGGTTTATGGGTTTATGGGTTTACCACCTCCAACTTAGAGTGGGTTTATCACCGTCTATCACCGTCTAAACTTAGAGAACTTAGAGAACTTAGAGCCACCTCCTATCGCTGTATCGCTGTGGGTTTATCACCGTCTATCACCGTCTATATCGCTGTCACCGTCTATATCGCTGTCACCGTCTACCACCTCCCCACCTCCCCACCTCCTCACCGTCTATATCGCTGTGGGTTTATATCGCTGTATCGCTGTCACCGTCTATGGGTTTATCACCGTCTACCACCTCCTGGGTTTATATCGCTGTCACCGTCTATATCGCTGTGGGTTTATGGGTTTACCACCTCCTCACCGTCTAAACTTAGAGTGGGTTTATCACCGTCTATCACCGTCTATATCGCTGTCACCGTCTACCACCTCCAACTTAGAGTCACCGTCTATATCGCTGAACTTAGAGCCACCTCCTCACCGTCTATATCGCTGCCACCTCCTATCGCTGTGGGTTTATGGGTTTAAACTTAGAGTGGGTTTATCACCGTCTAAACTTAGAGAACTTAGAGCCACCTCCTATCGCTGCCACCTCCAACTTAGAGTGGGTTTA'
    # k = 14
    # ans = frequentWords(genome, k)


    # pattern = 'CCAACACGTCGAAGTGAAACACGTCACCCTAAGCGGTATGTTCAGATCCCCAGTCACTCCGTGACTTAAGTGGACTAGGAGGTTATGCAGTATCGGGTGCATTTGGTTCCGCGCAATTTGGACATGACAATTGGGCTATAGGCTAAACGATAGTGCGTCATGCTCGAGCCACTCCTGTCCTAGGCAAACAAGGTTGGTTTTCCAGTTCTACAATTTTATCTGAGGGACCTGACCCATATCGTCGTTCACCGGATTTACCTCCATGTAAACGTCTGATAATGTAATAGCCTAGCTATTCATCACGGTCAGACCTACGGGTCAATAAGTGGGCCAACCAGTCGCCGAAAGGTCCCACTACCCCGCAAAGCGTCGAGTCAAAAGGGGAGCTCGCGGTTCAAACAATATGTTTTGTCGGGCTAACCGTCGCGTACGGAGGATGGACCGAGGATTTTCTTCATTCACAAAGGATTTGAACAATCGTTCAGCGGAGCAACACAAAATTAAATGGCTTAGCTAGCAGCTACGAGGCCAAACTAAACTCAAAAGATACCGTCCTCTCCCTGGAGTACGTTTGCCCACTTAGTTTATGTCTATAGCAATGTTTAGGCAAACCTGTAACTCTCGACGCGTATAGCAAAGTTGGCCCCATACGTACTTTCTTGCGCTTACGGCCTGGAGAAGAAGAGACAGTTTTGGGCATTAGAGCGTGATGTGTTGTTCTCGGCACAGGTACTGCGGGAGCCCTGCCGAGGAGTTAAAGAGAGTAGTGCTAAATAAACGACTAGTACAAGTTTTTTTCCGGCTAATACTGGACTCCTGCATAAGTGCATACAATGTAGCCCTCCACCTTTAGGGAGTAGTAATGTCAGCCAGCGGGCTTTGCGGTTGAAAACGTGGGACGTCCTGGTGGATATTTGAGAGCCGAGACGTCTAAGAATCCGTGCTACTTTCAGGATTTGGAACAGGAGGGGGTCTAATCGTGAGATGCGTCATGAGTAGCTAACCATAATATCTCCAGGGAAGGCAAGAGCGAGCGAAGTGTCCAGTGCTAGGTCACGGCTGTTACTTTTGAGCTGCTCATCGGTAGGCCGCCTGGCGGCGGCGAGCGAAAAAAGAACTGGCTTAACTGACTTGGCGCCGGAGAAAGTACCCTGGAAGTGTCGTACGCGGACTTCATACATATGATTAGCCGACTTCTCCAACGTAATGGCCTGACCCTGATCCAGCATCTCCCGTCGAGTATTTACCTGAAGGGTATACGTGCTTGTTTACACGAAAAAGCTAGACCTGACCCCGTAGGTACCATTTAATCGCACTATGATCCTTGCTGGCTTGCCACACAGCTCCTCTGCTTCATCCCACACCTGTAGGGTGCATTCCAGGAGTAATGCGTCATCGCGAATCATAACTAACGTCAACTGAGTAAGTTAGTACTTACTACCCGACCTCTGAGTCTCTCGCAATTTGCGGGGGTCCCGCGATATAACACGTGCAATTTAGCGACGTATGGGAGCATAATCTAACTTAACCGTACTACTATTGCGGGGTAATTGCGTCAAACAGTTAAGTCCTTGCAGCAAAGTTCGAAGCTTCCTGTTTCGAACCGGAAGGCAAACCGAGCACGTGCAAGCGGAAGAAGTTGGGGACGAACAAACATCGCATGCGCGAGGCGTAAGAACCATTATTTTCTGCTTTCGTTCGTTACAGCACATCTACCCGTCTATACCGCGCGCAATTTTTATCGTCAGTGTCTTGGGAACGTACTTATAGACCACCCTTAGAGTAAACGGAGCGCCCCAGGGGATTTGCAGCGGAGTAACGGCGAATACATCCTGTATGGTGGTGGATGCGCCGTGGTTACACTATTAATCCCGCTTAACCTTCGCTCGCTCGGCGGACTTGCCTGGTAGATGGCGTTCAACCTGAGCAACCCGGCCGGTTACCCACTTCCTCCGTGTTCAGGCATTCCTTACAGCAGAAGTGTTCGTCTTTCTCAGGGTGATCATGGTTAACGCGCTATAGGGCTAGTGCCTAGAGTGGGTCTGCAAGGTCTCTGCCGGCTTCAGGTATGCGGTATTTTTTATTGTCTCCGAGTTCAGCTATTTTGGGCACAAAGCGACCCTGCCTCTAACCGCAGTTGAATATACCGTTCTTGCTAGCCGCCGAAATTTTGTTTAACTTACTCGGTAGCACGACGCGCAATCTATTTTCTACGATCAACAACTCGGCGAGATCGTCTTCTCAGCGGTCGAATATATGCTTTCGCCAGGTCGCCAGGCAATGGCGCTAAGCCCAAGCGAACAATCAACCAAGGAAACCGAGGTTGACTGCGACGTTGCGGCTACGCTCTGCACACAGTATATTTTAGCAATCAGTCTGGAGCAATGCGACAGCATCCCGAACATATTTAGTGTACCAAGAGGGACCGTCACTACCAAACATTTCGAGACTGGGCACCGCGGGGATATCAATTTGGCAATTCCCCCTCCAGGCATGTTAATCAAATAACAAAAACCCCCATGGCCAGTCCGGCTCCAGCGGTTGCTGAAATGCGGCCGTCCACGGAAGCTTGAGTGACTGTGTAGGCTTACGACCGCCATCCGGTGTACCCTGCGCTGTTCCCCTTCTTCGCCAGGTCTAGAGTACCATGCCTACGGGCGATCTGTAGTCGCCGATGTTCCCGAAATCAATGTGCATCTATGCCCCCCGGCGTTCTAAATGTTCCCGCTACGTGGCACGGTACAGCAGACGTGGTAGGCCAAACTTTAGTTTCATGAAAGCAGATAGGCACTTCACTAGCCACCGGGGCGGGTATTACCGTCTTACCGCGGCTGAAATATAGTACGCCTATGCAATTTTGGTCTAGTGACCCATGACAATCCCACACTGTGTGGCTATAATGGACACATAGCCAGGGTCTAATTAGCCCCCCCTCGCCATTTAAGAATAAGGCCTGCAACGAAGTGCCCTAAGATAAGTCACGCGAACAGGAGCGGTTAATCACTCGAAAGCGAATGCATCCGCACCCGCTAACGCCCTTAGTCGACCTCTGCCCCGTAAAATTCAGTTACGTTCGGCTCCCCGTATACAAGGCCAGTGGATGCTCTTTAAGCGCACGCTCATAGATTAATTTAAAACTGTTTAAAACCACTTCTTATTCTTCCGTCCACTCAGCGACCTTAACGCTCACAAACTCGGCATCAACCGGGAACTGACCGACTGCCATCTGCATTGAGCTGCCACACTTGTACAATAGCTCCCGATTATCACCCCCCGATGATAGTAGATGGGCTGCTTAGAATCGATAGTTCGGCCCCACTGTTGGATGGGTATCCTAGGCAAATAACAGCTCCGCCTTTACGGCCTTTTTCGTTGTCGAGCCTTCATCTTAGCGGCCTGGTTCCAGACCCCTTAGAGCGCAACGCGCACCTCTTTACTCGCGTACTGGCTACCACATTATTGTGGCCATGTCGGTAGCTCTGTTCAGTATGTTACATTTAAGGTACCTGACATCTCCCGCTAACTTGGGGTTCTATCTTTAGCCTAGGCGAAAACCCAAGCCCGAAGACATTGTTGGCGGAAGTAAAGACACATGCGTGGTGGGTAATGCACTGGCCTGGCAGTGTACCTGGTTCACGTAATACACCCCCCGGAGCGGGCAATTCTTGAGTTCGCGTCATACTTTGTTCCGAAGGTAGCCCCCTCAGTGCGAAGAAGCCGGGTGTATAGCGGCGTGGTGGATCTGACTGTTCGCCACAGGCAGGGGAGTTAGATTCGTCCAAATACTTATATTACTATTGACAACTAGATTTTATTGATACTGCACCGTACAAATGGCACCCTGGATAATTATCCATCATGGGCCACCCATATGGCCATCAAGCAATGCCAGCAACCTCCTATTCAAACATAGGATTTAACGTCCTCAAACTATGACGTGACTGTGAATGGCTAGGGTCTGGACCATCTCGAACGGGAACTGACCACTAATATGATTTTCGCGTCTAGGTTAGTCCTTGTGACAAGGGCATCAGTTCTGTTATCATGCCTAGCATTAGGCACGTCGAAGTGACGCACCTTACAGGTGATCTGATTTCTATCGGAGAATACCAAGTGCTTAGAGCCCCAAACTCTTAAGAGAACCAATAATGAAGAACCGGTCAAAGGCGTATGAGAGAATGAACAATACTTCGCACTTGGGGTTATACGAACACGGGTGACCATATGAAGCCGCAGAGCTATTTGCGGATGACCAGAAGTGGGCGGTGGCGCTCAACTCGCTAACGTCACGCACTGCTGACAATAGTCTTACACACAAGGCTTCATCCAGAGTAACTGGATAAGCACTTGGGAGTTGCAAATAGAGCAACTACCGGATTCCGTAACTCTCACGCAAGTATTTCGCATTTTTACCAGACCCTGGAACATGCCACTCCGTGCAGGGTCACCAGCGGAGCAATCTATTCACCTATCCAGTGGGACCTGCACCATATACTGTCGAGGGTCCGGTAGTCGTGCTATGATCTCGATAGACTTCAGTATGTTGCTGATTTAGAAGTCGTCCACGAGTCAGACCTGTTAAAGGATCGGGAAATGGGCGGGACCAGTGTATTTAAGACACTGCCATATAAAACTTCCGTATCAGGAGTCGGCTTGGTGATCTTCGACTTTGAAATGCGCAGGGCTCCGCGGCACTGGCAAGGGGATCATACTGGCGCCTGCTACGTTGTCGCTAATTGGTTCTACCTTGCGGGTGTTGTGTTCCGTTGAGTGGAGTCTTTTGGAACGACTATGGGCCATCGTGTGAGTAGATCGCTCATTTCTTTCCGATTGCGCAACACCGGAATCTGAGATATCCATTTGCAAGCTCTAAGAGACTAGTTTATCCTACGGTTGCACTAGCCTCTATAGCGATCTGCAGCTTGATGAAATTTTATACTCCATATATGACCTGCATAAGTAATTCTGGGAGCCGCTGATACACAAAGTTTTAATTTCGCGATAGTTGTATTTGAAGCCTAAGGTGCCATGCTATGATTTACTGGTAAATTAGAGGCGTGGGGCAAAGTTTTCGTTGCAGGCGCGTCCCGGCCGAGGAAGAATGAATTCCTCCCATCGGCCTCTAATTGAACATCTTACCAGGCGTGCCTATATGACGACAGTAAAACCTTAGTTGACGAACACTCTAATAGATCTACCGTTAGCCAGGGCAGCGAAAGACATCTGCTGTGAGCACGACTGGCGGGAATTGACCATGGAAATATCCGACGCCTCCATACATGATTTAGGCACGAATCACGACCGTCCACATGGTATACACCCTTCGCGAGTCTGTAGCTCGAGCTCATCCATACACTGCCGGTGTGGTGGTAAGAATGACTCCCTGAAATACGATGGGTTAAACGAGGCGGTAACAAAACTCCATCCCGCTTTCTTCACACAACGTGCATGCTCTCGCTCGGCCTTGGCGACCGGGACTTATCACCCTGCGCCCCGAACGCCGCTGGGGAGAGTCTTCACCTACTCTGCGAGAAACAAACATCACCTACATACGGGGATCCACCTAACGATGAGCACCCGTCGGAAATAGAGTTAGTATAAGCGGAGAGCGACATTAAAACCGACCTTATGCCACGTGGGCGTATGGGAATCTCAGGCCGGATTGCCTCCATCGTTACTGTCCCATATGCTTTCCTGTGAAGGCCCCTTGGTTCTTTGCCGCTGTACGGATAGCGTCGCTTCGCCTTACGGTTTCGGCTCTTGGGCTTAGAGTCTTCTACGAAATGCCGCGGACATATCGTCGTACGGCTTTCAATGTTGTCGCAGTGTTAGGAGACCTCGTAGAGTCTCTTGACGGCGAGGAAAGATCTCAGACACTATACCGACGTTAGGTGGCGGCGTAAAGCGGATCGACGGAGGTTCTCAATGCGCGGCGTTCTCACCTTTGCGATCTATTCCGAGGCTGGTCTCTACGCGTCGGAGAGTTGCCGCTGCAGACCAACGTTCCGGTAGACCCGGTGATTAACGGCACACCCGGACTGGCATCATAGCAATATACTGCTGCGAGAGTGAATACGCTCATGGCGCCCTTGGCTGACAAAAAGTCCGTCGTGAGTACTATGTCCACAGGTCAACACCCTCGATTTGGTATGCGAAACCTCCGCAATGCTGCTGAGTTTTTTATCGGGATTTATCTCGACAGCGTCGTCACGATATTAGCAGGCCTGAGCACTGAAGATAATCGACACACGAGACCTAGAAAGACAGCCACTCCAAGGGGCCACCGCCATACCGTACTTTACATCAGGAAGGCTACAAAACTACATCCGTGACACAAACACGTTTTTAGCAACGGATCTCTTATTCGGCCACGTGAAGAAATCAACTATCACTACACAGACGTACGTCAGCGTACAAGAAGACGATGACGAGTTACCCATATCAGATTGACTGGCCCGCAGACATCCAAGGGGGATTAAGCCCCACGCGCACGCTGATGACATGTTATCCTCTGCTAGCTAAGTCGCTATACCTTTGGGGGCCGTAATATCGAGGAGACGGGTGGTCGATTCCTCCTTGGTTTATTGACCGGTACGTCGATGGTACGCGCTGCGTTTACTTATTCCCAACGAAAAAACTTTTGCCTAAAGTGATTACTTACCGCTGGGATCTCCCCTAGACCTTACTCACCCCAAGCTGTGTCTCATTCTTGATGCATCCCGGAGTAGCGTGGCGGCGTGGTTCCCATTGAATCCGCTAGGTGAATAATGTCTTTCGAGTACTGTAGAGAAACGTACTGGTATCTTGATGATTGCACCTTGCCAGAGGGTTCTGGGGCCTTCGTCGATGAGTTTCCTGCGGGAGGGCGCATATAGCTCAAACTTTACTATGATAGATGCCGGATATAAGTTCCAGAAAAATTCCCTATCTTCATCCCAAAGGTCAGTTGATAGCCTTTCTGAAAGTCTTGTACAATAATCGGCATGTGCTAGCGCGACACGCTTGCTGTAATCCGAGCGCCGCGTAGCTTAACAGAGACTTGTACAAATAAAACTTCCTAGCTCTGATAGCGTTTACAGTGTTCAGTCACATGGTGTCACAGAACAGTTTTCAGTTCCCCTGGACAGCTACGAGACCAGAGTAACAGCCCCGCTTGGCGCATTTCGAGGTTAGAGGATATACACCAATAAGCACAAAGGTCCCGGGCACAGCGTGGCGCAATGAGTGGAAAACTCCTTATTGTCATCACGGCCGTTGGGCTCATTCGGGGCTTTGTCGTTAAGACAAGACGTGAGCAATCGTTAGACAGGTAGACGATAGATTGACTCGGCCGAGGGGCGATGTTTATCCAAGGCATGTATTAACTCGAGATCGTCCTGGACGCGCGCTATCAGAAACACCTGCGGCTGTTCATGGTTTCGAGGTATGCCAGCGTAGTAGGTCTCACGGGTGACTCAAAACCTACCTCCTATCAAATGCGTCATGAGAACGGGTAAATCCTCGTTTGTTCGGTGTTGGTCTGACCTGTGACCTTCTAAGAATCGCGCAAATTAGCTACGAGTTTCTCGAACAGGCAGAGGGGATCGTCGCCTCCGCCACGTAGGAGAGTTTCATCCCGATCCAAATTTACCAGCCTGTCTACCAAACCCTCAACGCCAGAGAATGCAAATGTGCGCACCCGACTCAACGCCGTCGTTCGGGGAAGCGGTCGTAAGGGGAGGCTATTAACGGGGTGGCCGATAGCCAATTACGGCCAACAGGAGATCCCACAAGGGCAATGCGCTTGGAAACACTAGCAGGAGTGACACTTAACCGGGATACCCAGACTTACTTACATCAGCCGTCGTGACAAGACGGTCCCTATGCACTCCATCGCCGAACTTGAGGCATACCCAGATTGACACCTACGTTGAATGCTCTTCTCGGCTATCTATAGAAGCGCCAGTGCGGGTAGCCATGTGACTGCTGCCGAACGGGTATGATTTTGGCTGCATTCCTTGAGTGTAGAGTGTCTTTCACATAAAAACTCTCCACGGTAGCTTGATCCTTAGCACAAGCACAAGTCTGTTAAGACCGTGGTTGCCGTGAGAACGGTTAGGAGAGATCGACTGAATCTCTCCCAGTAATCGCGACTGCGCAGCGCAAAAGTTACCCTTATATCAATAACCCGAACGGGCGTCCGCAGCCTCCCAGACTGCGCTGAAAAAACAGAAAATAGGTTTGATACATCGGCATTAGCCCGTATATGTCGTCGCTGAAATCTTGCCCTTCGCACTCCGGTTAAAGTCGGGTCGGTTTCCCCAAATACTGGAGAAGGCAAGCGCAAATGCGACGTATCATTCACAGCAAACCCCGTAGGTTGGCGGTGGGCGGGTCGAAAAATCCATCATTCCGTTCATTTACCGGAACATCAGAAGTTGGAACGAAGGCCATATGTCATCTCAACGTCTGCACGTTATAGCCTCCTAACGAGCGTGACGTGGCTGGTTCGACCCTAGTGGAGGCCCGCACTGACTTCCGTTTCGGTAAATTAGCCCGGGTAAGTGGTGATCGAGACCGCGTGAGTCTTTTGTTAACGGACCAGCGGCAATCGGCGCCATTTACCTCGTAATCCCACTGCCCTGTAAAATGGCAACGACGGCGATCGTACGAGGTATGCAATCGCAGTCACCAGTGCCGGCACAGGTCTACCTGATTATACCGCAAGCTTTATTGCGTCACATCAGTTCCTTACTTAATTGAGGGCGGATCGTACGTCTGACATGGTTCGTCGCTTGTCCGAGTCGATAAACTAGATGTATGGTCCACAGCCCACCCCCACTTGAAAGGCGGTGCTATTCCAAGCAAGCCTCCGATTGCCTCTAATAAGAGCGCCGATCACCTTCTTGCGCCGCGCTGGCCTATTTCG'
    # ans = reverseComplement(pattern)


    # pattern = 'TGACGGGTG'
    # genome = 'CCACACAAATGACGGGATGACGGGCTGACGGGTGTGACGGGTGACGGGTGTGACGGGTGACGGGTGACGGGGTGACGGGTGACGGGTGACGGGGGGTGACGGGGTTCTGACGGGGCCGATGACGGGTGACGGGCTGACGGGCGTGACGGGGATCGTGACGGGCTTGACGGGGAAGGTGACGGGGTTGACGGGCTGACGGGGCGATGACGGGCCGATCTGACGGGAGAGCCAATGACGGGCGTATGACGGGATGACGGGTGACGGGCACCTTTTCCTGATGACGGGATGACGGGTGACGGGGTGACGGGCTCGGTGGTATGACGGGTTCTGACGGGCGCATGACGGGCGGTGACGGGTGACGGGGGAGTTAGAAGTTGACGGGGTGATGACGGGGTGACGGGCCATGACGGGTGACGGGTTCTCTAGTATTCTGACGGGCTGACGGGATGACGGGAATTGACGGGTGACGGGTGACGGGTGTGACGGGATGACGGGACGTATTGACGGGTCAGATGACGGGCAAAAGATGACGGGGATGACGGGTGACGGGTGACGGGATGACGGGCAATGACGGGTTGACGGGTATGACGGGATAGTTGACGGGTGACGGGGAGGTAACCGGTGACGGGGATTCTGACGGGAATTGTGGTGACGGGTGACGGGGTGACGGGACTTGACGGGTGACGGGTTGACGGGAGTGACGGGTTGACGGGTGACGGGATTGACGGGCTGACGGGCTGACGGGGTGACGGGTGACGGGTGACGGGTGACGGGATGACGGGTGACGGGCAGTGACGGGTGACGGGCCCTCCGTGACGGGGCCTGACGGGTATCGACGTGACGGGATGACGGGTGACGGGAGGGTGACGGGGCTGCCTGACGGGTTGACGGGATGACGGGCTGACGGGTGACGGGGTTGACGGGGCTGACGGGTGACGGGTTGACGGGCCATCTGACGGGTGACGGGTGACGGGTGACGGGTGACGGGCTTGACGGGTTCTGACGGGGTTGACGGGCCCTGCGAGCATGACGGGCGGTTGGGCTGACGGGCATGACGGGGGTGACGGGTGACGGGTGACGGGACGTTTGTATGACGGGTTGACGGGGTCTAGCGTGACGGGCTCAGTGACGGGTTGACGGGGAAATGTGACGGGCGTTTGACGGGATGACGGGTGACGGGCTCTGACGGGTATGACGGGGCCTGACGGGACTTGACGGGTGACGGGTTGACGGGTGACGGGTTTGACGGGTGACGGGAAATGACGGGGGTGACGGGGACAGCTTGTGACGGGATTTCCTGACGGGTGACGGGCTATGACGGGTGACGGGCGTGACGGGTATCTTGTGACGGGTGACGGGCTTGACGGGATGACGGGTGACGGGAGTGACGGGCTGACGGGTTGACGGGTGACGGGTGATGACGGGCGCATAAAGGAGTGACGGGTGACGGGGATGACGGGTTTGACGGGTGACGGGTGACGGGTGACGGGCGAATTGACGGGAGCTGTTTGACGGGACGTAAGCTGACGGGCGTATTGACGGGTGACGGGTCCACGTCCGCTGACGGGCATTGACGGGTGACGGGCTGACGGGTGATTGACGGGTGACGGGTGGATGACGGGAGTGACGGGCGTCCTGACGGGCTAAAACTGACGGGAGCATGACGGGTTGACGGGTATGACGGGTGTTGACGGGGGAGTGACGGGTGTCTGACGGGCATGACGGGCTATGACGGGTGACGGGAATTGACGGGTTCAGGCTCAGGACATGACGGGGCTTGACGGGTGACGGGATTGACGGGATAATCATGACGGGTGACGGGATGACGGGATGACGGGCGGACATAAGATTGACGGGTATTGACGGGAGTGACGGGCTATGACGGGGCACCAATACCTTGACGGGTCCGTGTAGGTTGACGGGCTACGTATATAGATGACGGGTGACGGGTGACGGGAGCCCGCACTGACGGGCCTTTGACGGGCTGTGACGGGATGACGGGTGACGGGTAGTGACGGGATAGTAATGACGGGTGACGGGAGTGTGACGGGTGTGACGGGTTCGGTAGCTATAAATTCTGACGGGTGACGGGCTATGACGGGTGACGGGGTGACGGGTGACGGGAATCACCAGGGTGACGGGTATCACTGACGGGTCATTGACGGGCCACTGACGGGTCGTGACGGGGCTGACGGGAGTGACGGGTGACGGGACACTGACGGGTCATTGACGGGACTGACGGGGATGAATGAACGTGACGGGCGTGACGGGCCCTGACGGGGTAGTGTTGACGGGTGACGGGCATGACGGGTGACGGGCAAGGGTCTGACGGGTGACGGGATGACGGGTAGTGACGGGGTTGACGGGATGACGGGCATGACGGGCCTGACGGGCTGACGGGGAAATGACGGGTTGACGGGTGACGGGCTCCTGACGGGTGACGGGAGGTATGTGACGGGCATGACGGGGCAGCAGAGATGACGGGATGACGGGGCCTTGACGGGACTGACGGGTCGACTTGACGGGCCCTCATGACGGGCATGACGGGGCGACCCCTGACGGGTGACGGGTGACGGGCACCTGACGGGTGACGGGCATGACGGGGTGACGGGACCTCGAGATGACGGGGTATCTGACGGGGTTTGACGGGCGCTGACGGGATATCACTGACGGGCTTGACGGGTGACGGGGTTTTGACGGGTGACGGGTGACGGGATGACGGGCCGATGACGGGTGACGGGTGACGGGTGACGGGTACTGACGGGTCCGTGACGGGTTCAGATGACGGGCAGTCTGACGGGTACGGGATGACGGGAACGCGCCTGACCATGACGGGATTGACGGGCGCTGCTACAACTAATGACGGGCCTATGACGGGGTGACGGGATGACGGGGCTGACGGGCTGGATGGCTAGTGACGGGCTTGACGGGCGCGTGACGGGATGACGGGGTACGTGACGGGTTATGACGGGCGCGATGACGGGGCTGACGGGTGACGGGGCGGTGACGGGGCTGACGGGTGACGGGAGGTTACCTGACGGGTGGCCTTGACGGGGGAGTGACGGGTGACGGGGCACCGTGTGACGGGTGACGGGTGACGGGTGACGGGTGACGGGGGCTGACGGGTTGACGGGGTGACGGGAATTGACGGGGGCTGACGGGAACATGACGGGCGCTGACGGGCTGACGGGTGACGGGCATGACGGGTGACGGGTGACGGGGTAATGACGGGCGCTTGAAAATTGACGGGCGCAACTGACGGGAGTGTCATTGACGGGTGACGGGTTTGACGGGAAGTGACGGGTGACGGGTGACGGGATGACGGGTGACGGGGTCTGACGGGTGACGGGTGACGGGTCCATTGACGGGGTCGTGACGGGCTGACGGGCGTGTTATGACGGGTGACGGGAGCTGACGGGTGCGGTTGACGGGTGACGGGTGATGACGGGAGCGTGACGGGCACGTTGACGGGCAGTTGTGGGGCTGACGGGTAGTGACGGGCCTTTCTGACGGGTACTGACGGGTGACGGGGAGCGTTGCTGACGGGTGACGGGTGACGGGGATGACGGGGTGACGGGTGACGGGAAGTGACGGGGATGACGGGGTGACGGGGTGACGGGTGACGGGACGCGCTTGTGACGGGTGACGGGCTTGACGGGTATGACGGGAACTTACCTGACGGGTGACGGGTACGAGATGACGGGGGTTGACGGGCGATTTAATTGATGACGGGCTTTTGACGGGTGACGGGTGACGGGTGACGGGTGACGGGGTGACGGGCTGACGGGTGACGGGTGTTAATAGGATTGCTGAATTGACGGGATGACGGGGGGGAGACGACTCGTGACGGGTCATGCAGTGACGGGGTGACGGGTAGTGACGGGATCGTGACGGGATATGACGGGCGTTAAACATGCTGACGGGTCTTGACGGGAACTAGTGACGGGTATGACGGGTTGACGGGGGTAGGTGACGGGTCAGATGACGGGTTGACGGGGGGTTGACGGGCGATTTTGACGGGCATGACGGGCCATGACGGGATACCTGACGGGCAATGTGACGGGTTGACGGGTATGACGGGTCATGACGGGACTTGACGGGTGACGGGGGCGGTGACGGGTGGTGACGGGTGACGGGGTGACGGGTTGACGGGACTTGACGGGTGACGGGTGACGGGTGACGGGTGACGGGTTGACGGGCTGACGGGTGACGGGCGGTTGACGGGAGATTGACGGGCTAATATTCGTCATGACGGGGGTGACGGGTTTTTGGTACCTGACGGGCCATGACGGGTGACGGGATGACGGGTTCGTGACGGGAGCCTTGCTATGACGGGTGACGGGTTTGACGGGGTGACGGGTTGATTCCTGACGGGTGACGGGATTTGACGGGCGTAAACTGACGGGGCTGTCCGTCTGACGGGTGACGGGTGACGGGTGACGGGAGATGACGGGATTGACGGGAACAACATGACGGGCAATGACGGGTGACGGGTGACGGGCTCCTGACGGGGTTTGACGGGCACTCGGATGACGGGTGACGGGTGACGGGGTTTGTTTGACGGGACTCGTGACGGGAATGACGGGTATGACGGGTGACGGGGTTATTGACGGGTCACTGACGGGTGACGGGATGACGGGATGACGGGGATATGACGGGCTACTGACGGGTTGACGGGTGATGACGGGGCTTTGACGGGTGACGGGTGACGGGATGACGGGGGGTTTCATGACGGGGTGCTTATGACGGGCGGAGTGACGGGGATGACGGGTGACGGGCTATAAGTGACGGGATGACGGGTGACGGGTCTTGACGGGTGACGGGTGACGGGGGTCGTGACGGGTTGACGGGGGGCTGACGGGTTGACGGGTGACGGGTGACGGGAATGACGGGGCGTGCTGACGGGTGATGGTGACGGGCTGACGGGGGGTTGACGGGCCGACGTCAGATGACGGGTCATATGACGGGCCTGGTTGACGGGGTGACGGGTGACGGGAGCCCCTGACGGGTGACGGGAAATGTGCCTGACGGGTGACGGGACCTGACGGGTCTGACGGGATGACGGGTCCACCTGACGGGCTGACGGGTGACGGGTAGTGACGGGTGCTACCTGACGGGATTGTGACGGGGGCCGTGACGGGCCTGACGGGCCTTGACGGGTGACGGGTGACGGGATGTTGACGGGGTGACGGGATCGGTTGACGGGTGACGGGCTGTGACGGGATACTGACGGGTGACGGGTCGTGACGGGTAGTGACGGGATTGACGGGGTGACGGGTGACGGGATGACGGGAGTTGACGGGTGACGGGTGACGGGTGACGGGGCGTTGACGGGATTGACGGGATGACGGGTTTGCATTGACGGGTGACGGGCCCCTGACGGGTCTGACGGGAGCTGACGGGGTGACGGGGATGACGGGCCACATGACGGGAGGTGACGGGATGTGACGGGTTGACGGGCCTTGACGGGTGGTTGACGGGTGACGGGGGCAGATGACGGGTGACGGGGTGACGGGTGTTGACGGGTACCAGTATGACGGGCTTGACGGGTTGACGGGTTACTGACGGGGGTTCTCTGACGGGTGACGGGTGACGGGCCCTTTAGATGACGGGGATGACGGGACCCTGACGGGGCATGACGGGTTTTGACGGGTGACGGGTTGACGGGAGTGTTCCTCGTGACGGGTGACGGGTGACGGGGTTGACGGGAGGATTCCGGTTGGGTGCATGACGGGATGACGGGTGACGGGGAGATGACGGGGCTCATGATGACGGGTGACGGGCACACTGACGGGTGACGGGGATTGACGGGCCTCTGACGGGATATAATGACGGGTTGACGGGCCATGACGGGTGACGGGTGACGGGCTGACGGGTGACGGGTTGTGACGGGTTTATGACGGGTGACGGGTGACGGGCGTATTCATGTGACGGGGATGACGGGTCTTGACGGGGTGACGGGCTGACGGGTCCTGACGGGTGACGGGTGCACAGCATGACGGGTGACGGGTGACGGGGCGTGACGGGATGACGGGCAGTGACGGGATGACGGGATATGACGGGTGGTTGACGGGTGGTGACGGGTTGTTGACGGGTTTGACGGGAAGGCGTGACGGGGGTGACGGGTGACGGGATAAGTTTTTCTAATTCAACGCAACTGACGGGTTGACGGGCTGACGGGATAATGACGGGATGACGGGCCCTGACGGGTTGACGGGTGACGGGACCATCTGACGGGTGACGGGACTGACGGGGTCCCCTTTGACGGGTGACGGGCGGTGACGGGGTGACGGGCTCGTCGTTGACGGGTTAACAAATGCATGACGGGTGACGGGTGACGGGTGACGGGTTGACGGGTGACGGGGTCGACTATATGACGGGGTGACGGGCTTTGTTGACGGGTGTTGACGGGTTCTTGACGGGGTTGACGGGTGACGGGTGACGGGTGACGGGTCACTGACGGGCACGCGCATGACGGGGTGTGACGGGTGATGACGGGCTGACGGGTTGACGGGTTGTGACGGGACTGACGGGATACATAGGCTGACGGGATGACGGGTTTCTGACGGGATTGACGGGCAGGTGACGGGTGACGGGTGACGGGTGACGGGGACTGACGGGTGACGGGTGACGGGTACTGACGGGTTGACGGGTTGACGGGCGATGACGGGAGCCTTGACGGGATGACGGGTCTGACGGGCTGACGGGTGACGGGAATGACGGGCTGACGGGAGCGTTGACGGGCTGACGGGTTTGCGCGTGACGGGCCGGGTGACGGGAGTGACGGGAATTGACGGGCTGACGGGTGACGGGATTTGACGGGGTGACGGGTCGCTTGACGGGTGACGGGTGCTTCTTGACGGGGCTGACGGGTGTGACGGGCTGACGGGAGCTGACGGGTGACGGGGTTAAGTGACGGGACTGACGGGCGGGCGTGACGGGCTGACGGGGTATGACGGGTGACGGGGATTTGACGGGTGACGGGTGACGGGGTGACGGGGGGATGCTGACGGGGTGACGAAAAGAGCTGACGGGCTCAAGAATGACGGGTGACGGGAATTTGACGGGCTGACGGGTGACGGGATTGACGGGCTGTGACGGGTGACGGGGGTGACGGGGTCATGACGGGCTGACGGGATATATCTGACGGGCTGACGGGCTGACGGGATGACGGGCCCGACTGACGGGACTGACGGGCATGACGGGTCATGACGGGTCTGACGGGTGACGGGTGACGGGTAATGACGGGCTGACGGGTGAATATGACGGGTACTGACGGGTATGACGGGCACTTTGACGGGAATTGACGGGCATGACGGGAACCGTGACGGGCAAAGTGACGGGTGACGGGCTGGGCAGGTGACGGGACTGACGGGATGACGGGCTCCTGACGGGTGACGGGGCCTGACGGGCAGTGACGGGCTGTGACGGGTCAAGGATATGCAGTGACGGGCATAAATGACGGGTTGACGGGTATGACGGGTGCTAATTGACGGGCGGTCTGACGGGATGACGGGAGTGACGGGTGACGGGATGACGGGTGACGGGGTGACGGGTTGACGGGTGATGACGGGTGACGGGTTTGCCTCATTTGACGGGTGACGGGTGACGGGTGACGGGGCATGACGGGTTGACGGGCTGACGGGATGACGGGTGACGGGTTGACGGGTGACGGGCTGACGGGCCTGTGACGGGATGACGGGATACGTTGACGGGGACACCTGACGGGTCTGACGGGTGACGGGGAGGGCTGACGGGAGTGACGGGGTGACGGGTGACGGGGTTTGACGGGGTCCTGACGGGATACTGACGGGTGTTGACGGGCAACAAGGTGACGGGCCGTTTGACGGGCTGACGGGACCTGACGGGTGACGGGAGTTTTGACGGGATCTTGACGGGTTGACGGGCATTTACCCATCTGACGGGACATTATCGGAATGACGGGTGAAATGACGGGTTGACGGGCATGACGGGTGACGGGTGACGGGCATGACGGGCCTTGACGGGTGACGGGAAGTCTGACGGGGTGACGGGTTTCCCTGACGGGTGACGGGTGACGGGTATGACGGGATTTTATGACGGGTTGTGACGGGGTGACGGGGTGACGGGGGTGACGGGTGACGGGGCGACAATATTTGACGGGTTTGACGGGTGACGGGAGGACATCTGACGGG'
    # ans = patternMatching(pattern, genome)


    ##### problem 1 #####
    # genome = 'GAAATCGAGTTGGCGTGTAACCAATCGCTTCGCGCTCCTCTGTAGAGGGGGCTATGTAAGAATTCGCAAGATGTTGAGCCATGGCGGTTTTTGGAAAGACAGGCGCAATTCCGTCAGAATAAGGAGCGGACACAGATTGTTGTTATGCCTCGCTCTGTCAGTCCGTGGAAAGGGCTTCCTTGCGGATTGGTACAATTCCCTGAATTAGATCCGCACGCCGCCCTCGCGGTATCCAGGGCATACCCTCCATAATGACAAATGCGTTGTAAAATGCGTTGTAATCTCGCGGTGGTTGCGTTGTATGCGTTGTACTTTGCGTTGTACCATTATTGCTACGGTTTGCGTTGTAAACTGGCTTCGGGTAATTTTAATCTGCGTGCGTTGTACCATGCAGTGTGGTGGTCGCCGGTGCGTTGTATTGATAATTTTGCGTTGCGTTGTAATTATATATTGCGTTGTAGACTATACCTGCGTTGTAGTACTAGCTGCGTTGTATACGGCTGGGGAACATTGCCGGCTCCCATGGATGCGTTGTACTGCGTTGTACCTATGCGTTGTAGTACCAATTGCGTTGTACCACCATCCTCCCTGGTGCGTTGCGTTGTAGCTTACTACCATTGGCGACGGCTATCCTGCGTTGTACGTTGTAACAAATCCTGAGGCTATGCGTTGTACCAAGACAACGTATCCCAAGGGGCAACGTATCACAACGGCCTACCGCGCGACCTTTATTGGATTGAACAGGTTCTTTGATTTGATATGCGTTGTTGCGTTGTAACTCTCACTGCCAAAGTCCTTAGGACCTCTCGGCTGCGTGCGTTGTAGCGTTGTTGCGTTGTATACTTTGGCTGGGTAGGTCGCGTGGCCCCCATGCCGCATCCGCACACTCCACTAAGCAACTGCCGCACCTCGAGTTATTGAAGCACGTTTTGTGTAGTTGCTTTCTTGTCTACAAGATTTAACGGCCGCAACGATTTAAGCTCCGGGGGTAATCAGGCATGTGGAGCACAGTGAACATTGCCCCGGACTCGTCTACCGGGCGGTCAATCCACTATCCTAAAGGACTCTGGGCATCAACACAGCGAACCACAACATGACATATGTGACGAGTTTCGAGCTGGCGATGGGTGTAAGATGTCAGATGAACGACGAATGGAGAACATGAAAATTCCCCTCAGACGTTAATGGTGTGCACGTACTGACACGCATGAGCTTGCCCCCGTCTAAGCACATACACTCTTCCTCGGCTCGTAGTGGTGATTACATAAAAATTACGCCAGTAAATCTTCTCACAGCCGTCCGACCGTAGTCTCCATCTACTGAGGCGCTTTCAGTCTCCATGAGTACCTACGGCACTAAAGTTGCCTGCGCGTCTCTGCCTACCTCTGCGCGTCTGGAGTTGCCACCATACATGTAATCGATGTCTCTGTCATTCTATAGCCATGATTGAATTTCGCTACTGGGCGTCCAAGTGTTCTCGTTAAGTCTTGAAGCCATCACTGGCGTAACGCAACGCAATGGTCAACGGAAGCAGGGGGCTTAACCTGCGATGTTAGCGCGATCACGGGTTCTAGGCCGCGTCGCTCTGCAATCGTCTGTAATCCTGGACGGTAGCAGAGATCGATACAACTTTTGACTTCTACTTATGTTCCTGATCACGTGCGGGGCACGACATCAGCCGCGGGATACGTTACTGAGTTCGACCGTAAAGGGCGAGGATGAGCAGAAGTACCCGGCTCTTGTCGAGTCGTTTCTTGCATCGAGATCAGATCGGATAAATAAAGCCTCTGCCTAACTTCCATTGAGCGAGTGTTCCCTATATGCGGAGAGTTGCTGCGGAGTGACCTTCCTGATCAAAGATGTCTCCATCTAAGGGTCCTGCCAAACGGTTGGACCCCCTTCCCCCTAATAGGCCACGTATATGTGCAGCTTGAAGCCTTGGCGGGTAATACGTCCGCCTCCTCAGATATGGCAGGTGGACAAATTTTTCCATGATTTTTTACACGGTTTATTCTAAAGTGCCGGCCGTCAGTTATACTTTGGCCCCGCGATGGCTCTGTTTTACGCCCGTGAGGTCATAGTTCGTCGGGGGCTCACGCCACGCTAACTATCACACACTTTACGAGGGCCACAAACGTAGCAATGGGCGTTTACTCTTGCGGCGACAGAGCATTGGTTAGCCTTCTTAATTACTACTGCAGCACGGAGCAGGCCCAGATCACCTGATGACACGCCTCTATCTGGTAGTAGGGTTAGCGTAACACAGATAAGCACAGGCGACCCCCTGCTCTTTCTAAGGTCTTTTGTCCGATGTGTTCTGCTGTCTGTAAGTGTCGGGGGGCTAGATGCGAACTATAGGATCAGTCAACCCACGTGGGAGCCCAGCCGATCCAGGGCCCATATTAACTCGTTATCTTGGCATATCTGGCCGTACATACAGTTTCATGAGAGTATCGACACCCCACATTATACCGCCACTCTGGCGTGAAGATTAGCTGAACAACCCGATTAGCTGAACCAACACTGCTAGCTGAACACTCCCTATAGCTGAACACTCTATGATGTATGTATAGCTGAACCAGCTCACATGCATGTACTCTCATTAAGATTAGCTGAACATGTACTCTATAGCTGAACTCTCTCGCTGGGCTCAGATAGCTGAACTAAATCGTAGCTGAACTACCCTGATAGCTGAACAGCTGAACAAGAATGACAGACATAGCTGAACATGTACTCTGGGTAGCTGAACGATGCCCTAGCTGAACGTAGCTGAACTAGCTGAACTCCTAGCTGAACAAATGTACTCTGTACTCTGAGAACATAGCTTAGCTGAACTACTCTTTATTGCATTACGAATATGTTAGCTGAACAGCTGAATAGCTGAACTACTCTTACTTAGCTGAACGAACTAGCTTAGCTGAACCTCTATGTACTCTTGTACTAGCTGAATAGCTAGCTGAACCATGTACTCTCTAGTCCTAGGGAAAGCGCTAATGTTAGCTGAACGCTGAACGTAAAAATGTACTCTAATGTAATGTACTCTTGCGCGTTACTTTGGAGGATCGAGGAATCAGTATCCATGTACTCTCCTTCAATTCGATAGTTCATGTACTCTTTCGGATGTACTCTAAGTGTGCACATTAGGCGCATTTGTTACGTCAGGCGGGTGTTGCCTCGAGCGCCGGACCATTCTCACACGGGGATTCTACCAGTCTCTGGTTCTACCGCGGACAAATGAGAAATAGGGCTCGCGTTAACTCGTATCACGGTCATTTTAACTAGAGTCCCTCGCTATCGAGCTCTAAACCAGCCAGCGCGGATCTGCAAGAGCCATTCAACACTCACGTTATTTTATTGAAGGGGAAACGATTGGTCCGTAGCCATACCGTCTCAGCCGGATGTAAGCACTCCGAGTGTTCAGGCACCCAAGAAGCCGTAGGATGTAGCCTGTTAATGGACAACCCCTCACAGCAGTTATCCGGGCATTAGTATATTTTCCATATTGAGTAGGACTGGTCCGAAGTTGCAATCCGGGCAGTACCCTGTATCCAGCGAATTGTCCCCTGTGGGAAATAGATTTCATACTCGCCCTTTCGCAGCGGAGGCGCCCAGGTACCCTGTGGGAGTGCAGGGCATCCTGTGCCTGTGGGAAACCGACCTGTGGGACCTGTGGGACCTGCCCTGTGGGAGAACCTGTGGGACTGAAGTGCCCCCTGTGGGAGTGGGAAACTCACCCCTGTGGGATTACCTGTGGGAGATAGCACAGGGTTCCTGTGGGAGCATCGTCGACGGTGCGATGTCCTGTGGGAGTTGGCATACGGCACCTGTGGGAATTGTTCACTACGGTGTAACTGTGCCCGTGAGGGTCACGCGGCCTGTGCCTGTGGGACTAGGCCAACCCCCTGTGGGAAACGTACCGACGGACCCTGTGGGACTGTGGGATAACCTTCAATTATGCCCTGTGGGATCACTATTCCCTGCCTGTGGGAACTGTATATCCCAAGAGGCTTGAAATCCAGAATGCCTGTGGGAGCCGTGTGCCTGTGGGAGCGTCATCCTGTGGGACTTCATTCTAACTTCACTTTCCCTGTGGGAGGACTGTGGGAACGTAACCTGTGGGAACCCTGTGGGAATCCTGTGGGAGTAGCTGGGGCCCTACCCTAAAGTTCTAGCAAGGGCAGTGCGCCGCCAGCTTGGATGTGCCCTACCCCAGGGCAGTCTTAATCGCGAAGGTTCTTACCCGTTAGCACCTCCACTGATGGGACTGCCGGTCCTTGATAACAGTTTCGTTTCGGGCGGCATCAACAGTTTCACAGTTTACAGTTTCGCATAAGACAGTTTCGCAGTGTAATAGTCGAGAACTTAAAAATAGAATCTTACACAGTTTACAGTTTCGGGACGAAAACGCACAGTTTCGAGTTTCGAAATCGCCAGTTACTTCACAGTTTCGGTTTCGACGCGGCGCATCCCTCTCACTTCACAACAGTTTCGTTTCGAGTTTCGGTGCAACAGTTTCGACGCACAGTTTCGTACAGTTTCGATTTCCACTAAGCTCGGACGGGTATAAATACGTACTGGGCTCACCAGGGAGCCTATGACAGTTTCGGCGTAGGCACCCGCACACTAACCCATGCCGGATGCACAGTTTCGAGTTTCGCGGCACTGCAGCTTCGCAAGATAATACTTTAGCACCGGACTTTACAGTTTCGCAATTGGCACAGTTTCGGGACTCTCAACGTTCGCTTTGGGCTCTGTTTTCCTGGGAACATGGTTTCGGGGACGGGAACATGAACATGGCGGGAACATGATAGCTACAGTTTCGTCGTTAGCAATTAGCTGTTAAGGGGGAACATGTGGGAACATGTTTCGGGGAACATGGTTAGCACCGTTATGAAACTCTGTCTATATGAGTGGGGAACATGATGGATAGCTCGCCAACCGTGTCCAGCAAGGGAACATGGCTCGGGCTGTCGGTATGACCAGAGCATCTGGGAACAGGGAACATGGGTGAGGCTAGTTAGGGCTCCGGCGCGCGGCGACTACTGGGAACATGTCCAGGGGAACATGATTTTTGGTACGGGAACATGGCCAGGGAACATGAGGGAACATGGGGAACATGGGGAACATGTAGCTCAGAGGGGAACATGACATGATCATGTTATTCCGAACATGAGGGTCTAGCCTTCTGGTGGGAGGGAACATGCGTGCACAGGAACCCATACGGGGAACATGCCCCAGTGGGAACATGAACATGGGATGGCTTGTAGGATGCCCGCACCACTACAGGGAACATGCGCAGATGGACCTATGTACTGGGAACATGCACGGGGAACATGCGCGAGATGCCCGCATGTCAATACCCACTTGACATGCCCGCAATCCCTGCACAGCCATGCCCGCAAATCTCTATGCCCGCAAAGCATGCCCGCAATGCCCGCAGTATGCCCGCAATCTCTCGAGCATTTCTTAATGCCCGCACACAATGCCCGCACCAGCATGCCCATGCCCATGCCCGCAAACAAAAATGCCCGCACATAATAGAGGTCAGAATGCCCGCAGGTTGTATGCCCATGCCCGCAATCCTGCGATGCTACATGCCCGCAATATGCCCGCACCGCAATGCATGCCCGCAATGCCCGCATGCCCGCAATGCCCGCAGAACAATCCTATGCCCGCACGCGACATCCATTAGGAACACCCTGCGATGCCCGCACCTGCGATGCCCCTGCGATGACCTGCGATGTGTCCACAATGCCCGCAAGCCTGCGATGGACGTCAACCTGCGATGAGCGTGACCTGTTCGGACCTGCGATGCGATGGCGTTTGTTCCCTATGAGAACTACCCAGCACACCAAGAGCTTCATACTAATACCTGCGATGACAGGTTCCCTGCGATGAGGGGTCTGCCCTGCGATGAGTACCCTGCCCTGCGATGGACCCTGCGATGGAGTTTTCGCTATACTCTATGTTACCGAGGGACAGAAATGTAAGGAGGACCTGCGATGGTGCCTGCGATGGATGCCTGCGATGCTAGATAGACTGCTTCCAGCGGCGACTGACCCGAACCTGCCTGCGCCTGCGATGTTCCATAGGCCTGCGATGACCCCTACTCAGTGCCTGCGCCTGCGATGATGGGGGCCAGGCAGATTCCATACTCGTGTGGAGATTGCACTTGGTGAGCTCATCCCCGAGTGGAACCGAGTGGAACAAGAACAGACAACGAGAACAGACGTCTAGAAGAACAGACTAGACTAGAGAACAGACAACCCAATGATTAGAACAGACACTAGTTCGGGCATTCTAACCAACATGGGGTGGGGAAGAACAGACAAATAACACTATCATATACATGCGCGGCTGGTAAGACAATTGCCTTTGTGGGCTGTCTGTACGCCTCAACGATACACTGGGGTCCTCATGTAAGAACAAGAACAGACTTCGGCCACAGAACAGACGAACCCAATTTGGAGAACAGAACAGACGAACAGACCCCTCAGTAGCGGAGGGAACGGCAGGCACACCAAGCAAGAACAAGAACAGACAGAACAGACTAGAACAGACCCAGTCAAGGCAGAGACAACACAAGAAGTTTCCGGTTGCATTAGGGAGAAAGAACAGACTGCCGACAGAGAACAGACAGAACAGACCTAAGAACAGACAGACTCCTATACGTACGCGCGCGCAGAACAGACCCCCGTGCTCGAAAGGCCGCCTAAGCATCCGAGAACAGACAGAACAGACACCAATGGGTAGAACAGACGCAGAACAGACAATAAGAACAGACACAGAACAGACGCTTTCTAGGACCCCTACCTCTAAAAGTTCAAGTCCTGGGGATGTAGAATGTGCTGACTGAGCCATGATGTCCCCCCCCCGTGCATATCTCAGATCGGGAGAGTGACTGGGTTCTCCGTGTTATGAACAAGTTGCGTCTTGGAGGGGGAAACAGATGCTATATAAAATTTGATGACGAGTAGATAGTTCCGAATGCTTCCTATATCAGGGGTCTCGATTCTGCATACGAGTGACTTGATATGCAAAGCGTCCCACACCAGGCACTCGAACATGCCGAAATTGGGTGCATTGCGATCTAAACTCAGTCGGAGTACTTCTCGTACTTATCGAATGGCTCATACTTCTGCTCTAGCCGAGACGCCGGAGGAACGGTCCCCTGGTTCGGGCTATAATAGTACATTGCTTTAGCGATGCAGCTCATGTACACTAACAAAACAGGTCTCCAGTGGTGGGCAGACGTACACTTAGCGAGCATTGCCTAGAGTGCAGAATAATGCTCCGTCAAAAACAATCCGGAATCCCCTGTAATGGAGGAGCTGACCCTTCGACGTCAGAGCCTAATTTAGTGGGATAAAAGACCCCGTAGTACACTTAGTCGTATGGGAAGGAGTCCGTGTTACACTTAGCGGTTTTGCGGCAGTCCATACACTTACACTTAGAGACACTTAGCTCAGTGAGTAAAGACAGTTACAGCCTACACTTAGCACTTAGTTAGAACACTTATTACACTTAGTTACACTTAGGCTAGATAGTACACTTAGTATACACTTAGCTACACTTACACTTAGCTTAGTACACTTATACACTTAGACGGCAGTCTATAGTACACTTAGCCCGAGTAGCGTCCCTCTCTTTCGTTTACACTTAGGCATCTGCAAATAAGATTCACAATACGCGTTCGGTCCGTTACACTTAGGGGGTAGTTTGAGGATCGACTGGTTCGCCTGAAATGTACACTTAGGTTGCATTACACTTAGTTATACACTTAGAAATTATCGATAATGAGTACACTTAGGTCTACTACACTTAGCTTAGTGAATTTCTGCCGCTCCTAGGGGTTATATAATTCAGGAAAGCCCGGTTATACGACCACTCAGCGCTATAGGTGGTTGGATAGGCATAGGTTCACAAACTCTCTGCTGCTGCGGGGCCTTGACAATATACATGCTTGCATCCGGGGATCCAGGATTGCGCTTTGTCGGGAATTTCTTCTAAAGGAGTCCTGACAAGGCGAACCAACATTTGCTTAGCCCAGTAATTTCACCACGGTGGCTCCGATTCAAACATCTAGACACCGTCATTGGCAGACGACCTGCCGGCTTTTCCCCGACTGGGCTTCTGACCGGGGTAATAGAAAGTTCGCCACCTGCGGTTGCACTTGCGATATACGGTAGGTTCGCATAGGCGAGTTATTCCCACCTACCTACACAGTGTCGCGAACCGCAAGGATGAGTCAAGCGAAAGGTCTGCGACTATCAGAATTTGTTAAGGTAGCGCGGTTTTTCGCTCGGCATATGGCACTAGCGGGGGTGATGGCGTAGTCCCCGCTCCTCGCATCCACTACACAAGTTAGTAGTATCCTAGTTAGGGGTCAATGAACGTTAACGACCCGCCAACTTACAGGCAAACAACCACTACCGAGTGAGTGATCATTTTCCGGCCCTGCGTTACGGTATTTTTCAAGTTGCCTTCTGGGATGCAAGCTTAACCTTTAAGGGCACGGTTTTGCTCTGGTACCCAACAGATTTGTTGCTACGTACCCCTTTGGTCAACATATATCTCGACTCATGGGGCGTTCCAGTCGTCTGCGCGGGTCACGTAACTGTTCTGTCCAGTTCCCGCGTATACTGAGCACCTACTAGAGCCCTTTAGCGGCACAATAGATCGTCCTTTTGTTCAATTTGTCTTCCAATAATTGGCGTGAGTCGCCATGTCCGATAGCTAGCTATCCAGTTATTTAGGGTGCGTAAACAAGCCTCTGTATACAGGTTACCTTACAGAGGACAGTGGGAATCAAATCTACGAGAGGCAAGCGCTTAAATCACACCTGAGCTTTGGCAAGCCCGTGATCATCCCACGCGCTTATCCATTACATGTATCTTCAAAATACTCATCTAGGTGCTGAGTGCCACCTGAATCTCTCGGGTTTCCTTTAAGATACCGGATCGTCAAGATGGACTTATAACGTGCCGTGCCGATGTCTGATCAGGCTTATGTCACCTACGTCCCAGACAGGTTTATAGTGATGTGGCGCTGGTGGGTAAGTTCGCGGCATGGATGGGACGATCCTCTGAATGGGTGTCTACCGTTGGACGGGTACTTTTAGCCTTCGCATGGGAGACTTATCTGATAGTTCGGAAAAACCCGCACTCGATGGAATGTCCGCAGAATTGTGCGCTTCCACCGGATTGACCTCTGCAAACTTATGAGTGATAGGATGTTGCCTATTAGTGTCGGGCTCAGGCGCTTGGTCTCGCGAGTCTTACTCGTCTAGACTTCTACGCTCATCCCTAGCACGAAATCAGCGATGGGCGCGCTACTCGAATACCCCTGGCTCCACTGGTTTTATCCCCGTAACTACCTGACTACCGGTG'
    # k, L, t = 9, 598, 16
    # ans = findClumps(genome, k, L, t)


    ##### problem 2 #####
    # genome = open('rosalind_ba1f.txt').read()
    # ans = skew(genome)


    ##### problem 3 #####
    # pattern = 'TCATCAGATTT'
    # genome = 'CATTGGATATTGGCTCAAGAAAGGGCCGAAGAATAGTTGTCTCCACCACTCCTCATCCGATACCACCAAATCCAGTGACGAGTCACTCAGATACACGCAACCTTGTGAAGTCTACCTCGCTTTTTTAAATGATTGCTCGATAAAGTCTCTCCAAACTTTGTTTTGTTGGCTGCAGATCAGTAAATCCCGGAACACGACGGATGTCAGGGTACTAAAATACTAAGAGTTTGGCTATAGCATCAATCTAGGTTTTGAACAGAATTCTAGGGCGAAAGATGTCTTGAATGAGCACATGGCTCAGAGTCGCCACGTGCTACATCTCCGAGCACAAACGTGCTGAATCTGCAGCATTACCCCAGCACAGATATTATCCCGCAAAGTAAATCAACCACACCGGGCAGAGCAGGACATCTTCCATGCTGTTACTATTCCGTCGATTGTTCTACGGAATGCAGATCGGCGTAACTCGTGTTCATCGCTAACTTGCTGCGCCTGGGGGGTGACCACTGCAGCCACTTCTAAACGGTGACGATTTAGACTTTAGTGTCAGTTGAAATCACCGTAGCGGGGCTCCGTTTCGGGGATGATACTTGGCAGTGCACCCGTTGAAATCACATAGGGGGGACTGTGTTGCAACGCGCCTACAGAGGACGCAAGCAATGGCAACAGTTCGCTCCGGTCCGAAAATCGCTGCTACGTGCGAGCATCCCATTTAGAAGCATATTCTCAGCTCCTCAGGAATTTTACTTCTAGCGTGTGCGAACCTCGAGGCAACGCTAAAGAATGAGTCACAGGCTGAGAACAGCATTAGGAGTTGTTCACGTTGCTTACATATATTATCCTTCGGTGACAATTTCTTGGGAATCTGAGGGCAGGTCAACCGCGCGAAGTCCTCAGTACCGCCCTCCGCCGCGATTAACGGTGCCTTCTCGGTGATAAGAGGAAATACACCTCAGGCATTGATACTCGGGGATTCGAAGTACGGTGTTTTTTGAATGTCTATTATCTATGTCGGCGAGGACGCCCGAGTTAATAATAGTCCAGTGATGCCAAAATGTTGTAGCAAACGGATGAGCAGTCCCACCCCTACTGGATTCCCGTGAGCAGGGAGTCGAAAAGTTCTGGCGGAACCCCCTGAAAACTTCTGAGCAAATTACCAGTGGGTTTACTCAAGGGGATCAAAAGCCCGTCGAGCAGGGACGCGCATCCCACTGTGCCGGTAGCATCGCTTTTCGGCAACTAGCTACAAGATAGTGCCGGGTACCGCGAACGGCCGCGCGACCTCAGTAGCTCACAGACGTCAGTAGTCACGGACTGAGTGGTATTCGCAAACTTCTCAAGAGATGCTCGGAAGCAATCTATTGGGTAGATGTCGGCGCCGTACGCGGGGACACAAGAATCGAGTAGGAACCCCCTATAAGTCCTTTCGGGAATGGAGCAATGCCGTAGCTATACCCAAGCCGGACGTCTCTGCCCTGTCATCTACCTAGATTAACGTGTGGAGACTCCATAGTGTGCCCTAGATAAGTTAACAAGTTGCCTCATAATCCGTAGGGTTAAGAAGGTTTCTTTCGACTACACTTCCCATACGCACACGCGGGCCGCCAGGACTCCGCCACGGGAGCCTTCCCCCTGACGGTCGAACCTTCCACTTCCCCTGGTCGGTTGTCGAAGGGACTTCAGTCTCCTTGCGCAGCCCTGTACCGTTGGTAAAGATCTAAGACAACGTCTGCTATCTAACTACATATCTGTCATAAGGACTAATGGGAGGACGGGATCATTGAACTAGATATAAGGTCTGAGCCCCCCGGCCAGACAGCAAGCGTTCGACCTGCACAACCAGTGAGCCACTTTGACATGATGATAGATGCATCCTGTATGTACTGGCTTCTTCGGACTGATAGCCATTTGGCTCTATTTCCTTACTGTAACTTATGGACGGTGAGAAGACCCATCTGACGGTTGCGTACCCAGCGACCCACCCCCCCGCGCAATGGAGAGCCATTAAGTACCCTCACCTTCCTCAGGCGCAGAACCGGACAGGGAGACGGGGACACGACAAAGGGACGCCATGCCGCAGTAACCATCGTCGTCGGGCGCCCGGCGACCCGTCGGTTATCGGCGCTAATAGGCCGCCTCAAGCCTAACTTTATAAAACGTATGGGGTGTTTAGAAGAGCGCTGATCGTGGGGCTTCTATATGCATAGAGACCTCACTGCTAGTTGCGAGATCAGCTAGCATCAACCCTACATATCTACGCCAACGGACACCATCGGCGCCTGTCCACGCCCCCACCTCCTCACAAAACAGAACCGGGACCGCTCACCGACCAGGACTAGCGCGTAATACTGATCCAGCTTCTCCCGGATGCTTCCCAAATATCCCGCTCGGGCGCCCGGGTGCACCTTGATCGGGTGAACATAATAGTTCTCGAGGTCCTCGGCAGCTATCATGTCTGTATCCGTTTCGAAGCAGTCACACATGGCGTCACAGTGGACCGAAAGTCTAGCTACCAACCTTTTGTGCGTCTCGCTAGTGAGACGTCAGTATCATAGTCTGAGTCGCATAATCCGTGGCTTCGGGTATACCGGTTATGCGCAGGGATATTCCCCAATCGGGAGTAGGTATCAAGGCAACAGATTCAATTCTATACGTAGGTGTAGCTAGCCTGAAATTCTCAGTAACAGTAGATAGAGGCATAGTTTTTTTGCACATCTTGAAACTGACGGGTAAAGTTTAGCTCAATTTGCACTCGCGAACCGATTTGTCACGGTTGACGACAGTGGCCACGGGCTATAGTCACTCTCGTACGGCATCACCACTACTCGATCGCGACTAATCAACCCCAGATCGTAGGGTCGTGGCTCGCGCTTATTTGGGTCCCTTCTCGGGAAAGACGCTTCCAGAGACCATCGAACGCTCACATATTGTATCAGTGCTAGAGCATAGCCCTAATATTGAGGCTCGTCTACCGAGGATGTTTTCGAGCCGGACAGTATACACTAGCAACGAGCGGTTCTCGGATCTTTAACTAACCAAAAATCATAGCACTTAAACTGCGTCAACTTGCGAACTTGGATAGAAAACCTCCACTACTAGAGTCTCTTTTATAGTGAAAGTTTTTTTTGCGCAGCTACGACGCGACTTGGAACACTTCAGTCTATGTGAAGGCAACGGACAGAATCATGGGGGTTGGTCGCTCGAACTCCCATAGCACCAGCCTCAATTCGGCTACTGTCGTATTTCCTCGATAACAAGAAAGTATTCCTTATCGGCGCGCTTTGCTGGTTGCCGACCGTGTGTATTCGAATCATAAGGGTCGTTAGGCCACACGGTGTAAGTTACTTCCAGGGACGTCATACACTTAGAGTCGCATCAGCTTTCGTCAACACCGACGGCTAACTACGTTCTGTGGCGACTGACCTCGCGGTGTAATCTCTCCCGCAGAGCTAAAATCCCGTCCGTCATTGTTTTTTCACAGGAGTAGAAACCAGTAATACCGGCCTAAGTGGCCCTTACCGCTCTAATCAATTGTATACGGAAGACACCTCGTCAGCACAACACCAACTAGACCGACTTGGCTGGACACCAGCTCCTGGACGGATCTGGCCTATGCTCTTTTTCGGTACCGACGATAGAGTTCTTGTGCATACTACCGCCTTCATATCTGGATTTTTCAGCCGAGAATACAGCTTATGGTCTCCTGGCTTCAACTGAGAAGATTGGTCTATGCAAAGGCAGTTTCTCGTTAGGTTGCCGTTCTGAGGGGTATGGGGTACGCTCCGATAGATCGGTACGGGGTCCATTTGATCCACAAAAGAAGCGCAAGTATGCATGGACACCCCGCGGCCGCGCATAGACGTGCATTAGAACCATTCATTGTTGAATACGCGCCTGGAACACCATGATTCATTTGATTGCACGTTGCCTCTCGTCAATGAGTGAGTTGCGGTTGTTATGTAGCAATTTCGGCTACGTGGCCCAGCAGGCTGGTCGTTGCTATGCCGTCAGCTACGAGGCTGACTACTGACCCTGCGAAGTCTTCGTCCCTCAACCCAGGGCTCAAAAGTATGACCGCATCGTAATCGCTAAATCAGCAACACAGTGCGGCTTGGGTGTCGCGCGTGTCATCATGTGCGCTAAAGTTCGGCAGCACCAAGGTTCCTAACAAAAAACGCACATGATGAGAAGTGACTGCTAACCTGTCGAAGGGAACAGGCCAATCGTTCGGTCGTATTACTGTAACTAGAATTACAAGCTTCTGGATAGGCAGTCATACTTAGCTTCCCTGTTGTTGATCGTTTCCACGCAGATAACGGTCTCGGTAGAACGAGTTTATTGAATAAGGCTCATAACATCAAGGCGTTTTTAGGTGCAACTCGACACATGCGTAATAAATGGGATGCCGAATGTCCAGTCCACGTCAGGCGCCATAGGATGCGTGCAAAACCCATCTGCTACACGTGTAACATCTCAAATATCCGTATTGGCACCGTCAGTCTGAGAGGGACACGAATTACTTGCCACACGGAGTTCAAGCGCGGGACTCAAACTCTATATGAGCTACCAAACACCCCCCGAGGCGGCATCCGAGAATGACATGGGCCATCGTAATCGTGGTTCGTTAGCGGACTTGCCAGGGCCTGGGAGCAAGCCTTCCAACCTTAGACTGAATGTGTCGCCGCAGATTAATATTGTGATCACTTGGGCCTTTTTACGACTAGGGGACCACTAGATTAACTGACTCGTGTCCAAAGCACCACGGACTCAGTATCGTGATAGGGCTCGGTTACACCTAGCTGAACAAGGTTTTGGATTTAACGAACCGAAGAGCGGGGTACCCTAATTGAGGACGATGCTGTGCCTTGTAAACATCGGTAGGACAGTCTCAGAATTAATGTAAAGCTATATGGTAACGCTGGTGAGTGCGTTACACCTGAATCGGCCCCTGTACTACACCACGCCCACCCCGCACAGGTATCCGGGACGTTATCGCCACGAATCGGGCACTTGTTTGAGCAGCGCAACTAACATCGCGAGATAGTCATTTGATAACCGTAGGATTTAGATCACTCGAGCGAACCACCTTTGGGCAGCGGATGGACCCCAACGTCAAGTGGCTGCCAACTCGGTTGATTACGCGTTGGGGACTCACGTGCACCGCCGTCCACCTATCCGTCGTCCTCGCTAACCCGTGTGCACACAGAAGGTTGCGAATGTACTGGGTCCCACCTGAGGGTCCGCCCGCAACTTTCTAGACCTACCGCGCCAACCTCTTTGCGAGGTCAGTGGCACATTTCCGGTGCTAACACCGGGATCAATTAACCTCCTTAGCTAGAAGTGCCAATCCATCCACGAGAAACTTGCTAGGGTCTCTTCACAGTGCCGTATGTCCGCACTCACCACCATGTTTGTACCTCCTCTGCTCGGTGAATGCTCCTCGTGTGCACAATAAATCCCTATGCTGATCTAAGTTCAGTTCGAGACAGACGCTCTCAACCTAACGTTAACGAGGTATTTACAATTGTGGGCGGACAGGCGGAGCTGCTAATTACTCTGATAGTGTCTAGTCGTACACGGCATGGTGTTCATCGACAAGGGCCTCGATTGGAACGGTAACAGTAAATGGTTGATGCACTCCTGTAAGGCAGACGCTGCTTCTACGGCCATAGGGTGCGAATAGTTGTACATATCACGAGATTATCTAACGTGCTAACCACCGCGAATTTGGATTTATTATGCGTGTATCCACGTGATTCAAGCGAGGGACGATGAAGCCCGGAGCAATTGCTGCTGTGCCTTACGGGAGACCAACCCCGATAGATAAGGTGTATCAGACTCGCGTCAGGTAAATAATCCGTTTTCGCGTTGACCAGGGGATACCATGGGTGTATTTGCAGTAAAGGTTAGTGATCTTTCGTGACCGACGGGATGAAGCACTCCGTTGGACATCGGGCCAAATGCTACCATGGCAAAGCTAGCGTGACATTTCGGCCGGCGGTACAATAACAGATTGTACACGGTATGTTTTAAGCATCAGTCCTGGTTTTAGAAACTTTTTTCTTCTCATTGTCATATCGAACCATAGGTATAGGGGTCGGAATCCCCTGGCCGCCTAATCAAAGGGAGTTGCCCCTCAAATTTTACAAGGTCTCTAACGCTTATTGCGCCGTGAAGTACTGTGTAAGCACCCCGTAATAATTCGCTAGACTGCGGAATGTTGGTTCTTCACCGGCCGGTTCACACAAGCTAGCCGTCGACTTCCCGTCCCGGCTGTTAAGCTCAATACTAGAAAACTCCCTCCTCCTGCCTCACGGTGGTGAAACAATACTGACGTTCATGAGTTCACTCGGTGGACGCTGTGAAGTAGCCTACTGATCAATTGTGGACGTGCCGGTGCATTCGGGGATGGCGGAGTCTTTTATTCCGCGGGCGGAACTGTGCACGTCGTTCGCAATCTTTCGGCAGGGTCTGAGCCATTCTGTTGCCATGATAAGAAAGGTGTCAGGTTACCGCAGAGGTTTATACTCACGCTGTCTCAGGTCGTCTGACGTGTCTTACTACACCCAGTCGGGGCTGTCCCGTGCACATGCCAGGGCTGGGCGTACATCTCCGATAGCATTGGACCGGTGAGGCCCTGACCGGGGGTTATTGCAATCCCCGCTTTTCCATGACAGATGGACCCTTGTAACACTCTCAACCAGGAACCATGGTACATGGCTATCACGGCAGGGGGGCGAAGCTTAAAATAGAGTCGGACATCTCTAGATATCCTGGAGCCTCGCAGAGATGTACCCCAAGGCCGCCGGGGATTGGAGTGGGGAAATGGAGTTTACCTCTATAAAAATGGTGTCACTCTCATGTCTATGACTCTAGTACTTTCCCGCGAAGGGTTGCTCGGTCCATGGACCAACGCTAAATGTTTTCTTAACCAGGAAACGTATCGAGTATCTGGGTCTTATTTATATGATGTACCGTGGGCGTATTTACGCTCCCTGATACCAACCGCTTGAAATCATCCTACGATACGCTAGCAGCGTTGTGGTTGTGCCCCGACCAGTGACTGACCCGGCGAAGCAGGGTTGACTGGTATCGGATTCCTAATTCCGGATCGTAAAACCCCACCCCAGGCTTGACGTTCGACACGAATCCTACAAGACTCAAAAGAAATTCAAAAAAGGGGCCTGACTGCAGTCGCAGGCATCAAAGGGTGAGAAATAGCGCTCAATCTCACCGAAACGATACTACAACGCCTACGATCTAAAGGGAACTAGCAGGTTTGCCAGTGTCGTCTTGGTGGCCTGATCATGTGAATCACGCCATCTCAAATAAGACAGGGAGTGATAACTCATCCTGTGATATTATCTGAGATGTTTTCATGAAACACGACGCCTTACAGCTCGCCCACGTTATGTCAGTTAGATCGTCCACCTTTATATATAGACGACAGCTAAATTGGTATACCTTGCGCAAGTCCACATGCAATAGTTAGGTCCATCGCTGACATCACCAATACCGGACGAGCCCGGGGCGAAAACGCCGTAGCATTTTGCCGTAATCAGAACCTCGTACCATTGCTACATTTGGTCTAAGATTCCTGCCTGCAAGACGGGGTTACAAAACTATTTGTGTGGTTTTGCGGGGGGAGAAGCGAATTAGGAATGCGTAGGTGCAACTAGCCGTGCCGTGCATGCATCATCCACGTGATGAACTGCCGTCAACATCGCGATTACAACCATTCCTGAGTCCTTAGAACCCGTAAGCCCTCAGCGTCAAGTCGGGGCTTAATATTGTTATTTAAGCCAGATAGGAGAATATGTAGCCTAATTGAATAAGTCTACCAGCTTGCGGGATACACGCCGTTAGCCATGAATTTGGTACAAAATTGCTATTTCCAACCAGAGCTCGCACAATCTGCGACCAATACTATAAATACAGCTGGCAAGTTGATAGGTGTGCACTAATGCTTGCAGCTAGTGGAATTACGGTAGACTCCGTCTGATCCGGGTCAACTGATACGGACAACCGATTGCAACGTGACCCGCCTTCAAACCTTTCAATGATGCTATGATATGTGTGCTGATTCGATGCGCTCATTTCCACATCGGTCTCTCTGGGCACAGTCCTACTGGTACTGTATCAAAACTAGGCGAACCAGGGTCCCCCGCACTGCACGTGTCCATCAAATGCGCGCCCAGTCTGAAGGATTCGAAGGGCACTGAATCGGGGGCGACACCTGAATGGCTGAGAGCTGGATGGCACCCACAGACAGGCGTTGAGTCAATAAGCAATTCCCGACGTGGCGTTGCGGTTGCCTATCCAATGGACTAGTCAATAGCATATACTTCCGGGATTCCGGTTATGACGACGCAGCTACTCTTATACTTGCAGCGAGTCTGAAAAACGGAACAGCGTTCAGGTTCTATTGTCAATCATCTGAATTAGTCTCCACCTCAATAAAGAGGAGACCCCCAAAGTCACCGACGAGATACTGAAAGACCGTGTCGGTCCACTGTCCTAGGAAAGCGGTTAGTTTGTCTCTGCATTTATATTAGCCGGTTGCGGTAATTAGTCTTTCTCTGGTGTAGGCGGCTATCAGCAGCACCACCCACGCCTAAGGGCAGTTGGCCGTTCCTGTGATATGACCGCCGGGAGTCTTCTCATAGTCTCAAGGAGACCGAAAGAAACCACGGCCTGGACCGTCGATCGAATGGGGGCCAGGAACGAGGAAGAAATGCCGAGATTCCCACTCTGTGATGATCACGGTCAGCGCTCTTCAGGTTACGAGTTTGCAATCGCGACACACAATCTCTGAGACATATCGAACCACGGTCTTTTTTCATCTACCTAATAAAAGAATTGACCACTGGGGTAATTTTGAATCATCCCTACTTTGTAATGTGGATCGGTCCTAGACACGCAATACTGAGCGGATGTAGATGCTGGCCTTGGTGCATAACTATCCCTCCCCAGCAACGGAAAACGAACGTAAGCCGGGTTATTCATAGATAGCACTCGTGGGCGGAACTCACTGTTGACTATCAGAGGCGACGTGTATACGTGCACGCGAGGTTATTGCCGACTCTACTGTGGTATAATCTCCTCCGAATTATACCAAGGAACCAGATTGATTTCACAGAACTAGATTCAGGATTGGCCCCACTGTACCTTGTAACGGCTCACTGGCTAGAATTCTTTTCGAAATGGTGATCTGACACTTGACTGCAGAGTGGTCTCGCTAGAGTCGGTCATAAGTCGATCATAATTGCTTTTGATGCACTGGAAGAGTATTTTTTCAAGGGGAGAGGTCCGGTCGATAAGCCGGTGTTCTAGCTCCTACCGAGCAGAGCGACGGGCAATGTGGTGGCCTCCGGGGTGGCTGCCCTCGGTAATTCTGTACATGTTAACTCAAGGCCACTCTAGAGTACATACTTTACTTCATGCAACGCGAGACGGACGCCTACGACTAACTACAGATTTTCTCATATAAGGCGTACGAAACGCCACAAGTTGCAATAATAGGGATGCGGTAAGTGTCGGTCCTCCGGAACTGGCTCGGGCACCAACAATCAATTCCCAACCTACCACTAATAGAACAATCACCAGCAACTTCAAATAATTCGAGTTATAGGGAGCTTATCTATTTACTGAGCCCCCTTAACGAGAACTTCATGACTGGCGTAATGCTGGCGAAAACACACCTAAACCCAGTGATGTTTCAAGAATGTGAATGTTCTTTTGCGTCTCTTTAGTCTACGCCACGTACCCGCGCCCGCTAGTTTGACATCGTCTTTCAGAGATTTGTTTCTATCCGCCTAATCTTGGATTACAGACTTAGCACTCAAGGCTCTCAGGCTTAAGGTATATGAAGTAACACACTCATCCTAATGTTAGGACGGCGCTGTCCGGCCAAACCTAATGAGAAATCGGGATATCTTTATGTCAGAATATATGGACATTAGCGGTCCAAAGGTTTCTTAATTCACTATTGTTACGTGTTTAATAGGCAAGAGATTACCAGACAGGTGCTGGCACTCGCGATCGCCGCCGGGTTTGCCCGCTTGTGGTAGGAGCCAGTTACCAGATATATCGTAGTTAAGATGTACTCTTAACGCGCCGCGCTATGGTTCGACAGAGGGGAAGTTCCGCGCTAGGGCTCAGCAGCTCCGACCCCGGTAGGCGTCGTGAGCCGAAATTTAGGGCAACAACGATCGGGTCTTGGATAACCGTAGCCTGTCCAGTCGTAGGAGACTTTGAAGTAACCCAGCGACCCCCCGAGCCTTATCGATTTAAGGAGCCGCTGCTTTTACAGTTAAGTGGCGGGTTTCTACATGTAACCTTGGTAGTAGATTGTGTCGACTCTATGGACAGGAGCGGTCGATTGTTGATACAATATTGGTCGTATAATCCGTAATCGTTATGTTCAGCGAGGACTTACTTAGAGGGCAAAATATCGCCGCGAGAACTAGACTCCTGGCTATCTATTGCTTGATTGCTCGCGCGTATTGGAGCCACCGTAGGTGTTTACAAGATGGTGAGGTCTCATGTATCGACTGGGCTCCCGTGAGCCTGGTACCCCGCAGATAGCTGTTAAGTTGGAGATTTGTTGATAGCGAACATCAGCCGTGCTGCGGAAGCGTCCGCAGTAAAACCACTGTCTCCGATAGTAGGTCAATCAGAAACGGATCCGAAACGGAAATGTACCTTGTCAGGGTCCCGTAGAATTAAGTATAAACCCTAGTTAATGTGCACTGGAGTGGAACACACAGTGGGTAGACTCGCCGCGATATTCTGACACGTCGAACTAGTCAGTAAGCGGACATCCGGCAATGGCCTATGAACAAACGGAGGTTTTGGCAGGGCGGGAGATACCCGTGGTAATGCCGGCCACCAGATAACTGTCCTCAGGGTTATACACCAGCCGTGCTTGAGTGGTTGTCATTAATGGAGAGTGCTTCCCCAGAACTATCGCAAAACCGGGCCCTGTTTCAAACCAATTCCACGCCTACGCTTTCACGGGATACACGGTGATATTATAGGTCACAGCACGCCTGTATAAAAGTCCATACATCTTTCCCGTAGTCCAACCGTCTACCTAGGGCGGGGCTTGGCCAACCTATAGTACCAGTATACGGGAAAGGCTGGGGAAGAATACGGATGGTTAGGTCTAGTATGGCCCCGAGTGCTGCGTAAGTTCCGCGGGACTAGCTTGAGGGATACCCTTCCCGCTATGAGCTGCCGGTTTCCAAGGGTCATCTAAAGTGTGATCACACGAGTTAAAACGCCGTTGCCGAGGGTCGAATTATTGACTTCTCCGTGACTTAGATTATTGCAGATCTGGCAAAGAATTGCAATACTCCCGATGACTCTAGAAAATCATGTCAGCGCTGTGCCGACGGCGATATACAGGATCTCCACCTCAGAACGGGCGCAATATACCCCAGGGTTCTAAACGCCAAGTCTCCTGGGAACAATAAAGTTTAACGAAACGGATGCTCATAAGACGCTGATTTAACATACTCTATAAAGATACTTTGAGTCTCCCACAAACGCATCTTTCACTTATCTATCGATGTATGAAAAACCTAAGACTATAGAAGCTTTTAGAACGCACACCCGCATCTCTGGTTACAATCGTCGTCCTAATGCGACTGGTCACCAACGCTCATGAGCCCAGCACGATTTCCCCCAAGTCGAGAGAGCCAAGACCTATCGCAGCAGTCACTCGAATGAACTTACCCCGACCTTGCGTATGGCGCTTTATTGGATAGTAGCTGTTCGTAAGGGTTGTAGGCCTGATTCCTGTCGACTGAATGTCGTCGGGGCGCAATATACCTTGCCAGAATCTCCATCTTTGGGCGGCCGTTGGGAGTGCGTACTGAGAAACAGTGTTACAGTTAGTACTGCTGGCCGCTTGGCAAGAAGTCGTGATGGATCGGTCCCTACTCAACCCAAATTTTGTATTCAACGTGTGAAGAGCTACACTACGTCTGGGACATCAGTACCATGTTAGACTGCATGTGGGCGACTCACTGCCCAACTACGGCGGTTTGCAGCCGGTATGAGGGCCAGTGACGCCCAGGGTTAAGCTCGATAGCCCTCCTTCACTGCAAGTCGGCGCCGAGCCAACTTTAACGAGGACATGGGCCGGCCCCCGTCGTATCGCCAATCGTACAGCATACGCGCGGGTAGTTACCACGAAGTCACCCCATGGCGGATATTGGATCGGTCAAGTTTTGATGGTAGATGTACTCTTGAGGGCCGAAGGTCGAGCATCTAGCTCCCTGCACTTGTGAGCGACAAGGCCCAATACTTGGCAGATCATCGCAGAGATCAGAATAGCGTTCCATAGTGTAGTATCGACCGCACCGACCTACATGTTGCTTCACAGGGTGTGTTGCTCGTTATTCGAAGTTGAGTATTCCTCTTTCGTTAATGCAGCCAAGTACGCTACCCTAAATGTCTTCCCGCGTATTCGAACAAGCATCATTATAGGCCCAAGCTCACGGCTTCTCAACTGTGAGATTACTATCTAAGCCTCTCATTCACCATTGGTTTTCGCTGTAGTCCACACATAGATTCGCACCACAGGTCTCATGCTTGTGCCGGAGACCAAGTGCGATACTCCTGGTGTCCGATGCAGGCTGTCGCGGGTAGGCGGAACTCCGATGGCGAGCCGGTGCGTGCAGGTGGTCGGCTTCCAACTGTCTCACCGTCCGTTATGTACGATCTTTAGTTTGAGCAGTCAAGATAACCTGGGGAGTTTAATTACCCCAGCGAGTGTGCGCTTCCCTCCTGGCCAGATGTAGGACAACGATGCCCTCGAGGATATGAGATTTAGCAACGCGTATAATTGACGACCATAATCGACCCCAGACACTGGTTCGAAGGCCGGCACCTTTCACTCGGTCATAGCGTTAGCCGATCCAGGGGAGGCATCCGAGTGTCCGAAAGGGGATTCGCGGTACCAGGATGGCACACTAGTAGTTTTACTTCTTTCTCAAGAGGAATTATTAAGAAGTTCGTGAACAGGCCCTTCCAAATGGCCTGTAGTGGCTATTTCGTAGGTGCCGAAACAAGGAAGCCTTACGGTCACTCGCCCGGTCCTGAACGAGACTTGAGCGCTAGGGTTGCGACTATCAGTAGAACTTCCTATGAGCTCTGGTTCAAGAGTTCGCCATACACCATATAAGCGTAAGAACCTTTGCAAGCCCGCTGATAACTAAGTCTATACCGGCGGAAATAGGGATTTATCGTGTATCACACCAAGGAGTTAGCAATATAAAGCATTGATTAATAGCCCTGAAGCACCTGGCTTTCAGGTTATCCTACCGATTAACGAAACAAAGGAGCATAGTAGTTGCGAGCTTGCGCATAACATCCCTCCGATTGAAGTGGGAGTCATGTCGGACAGACGGCAAAACGGCGGGCACTCCCAAATGAGAACGCGCAGAGGTAAGTTTCGTCGCCGCTCAATTATCTTGATGGCAAATTTTTCACCGATTGTCGACGTGCGAACTCCTAACTAGCCTCTTTATATTCTCCCTGGGAAAAATGCTTCCACATCTATTTATATGATTGCACTCTCTCAAAAGATTCCGAAGGAAGCGTAAAGCCGTTATGATGTTACACTTATATCTTTTAGATGGTGTCAGGTACGAATATAGGCTCGCAAGTAACTCTACGCCTTTTGCCGACCCTGTGTCACTCTCCCTCCTCTGCGCTACAGTGATCTTATGAGTCCGTGCGCATCCCTGATAGTAGAGTGCTGGACGACCGTGTCTCGCTAAGGGCGGCAATGACTCAGAATTACCGCCAATGAGGTGATGATTGAATGTCCTTTTTCTTACCGGCCTATGGCCGAAGGACCCGAGATCCCCGAACCTTCATTTGTCGTAGCTACTATTGGGTACACGGTAGGCTAGCCGATTCATCTGGGTGCGGTCCGAGCGGAGATCATACACTCATTTAACGCACCACATGCTCGCTTTGCGACAGATGGGCTTATCATTTACACAGGAAGGTCAGCGGTCGATAGCCCTAGAGACGAGATACGTATAGATTTGTTCTTTGGTTTCAGCTCGTCATGTTGCCTCTTCAGTACAAAGTGACGTACCGCAACCACCTCCATTTGATATGAGAGATTCTTTTAACGGCAAACTACCGCTGATTTCCCGGTTGTCCATAGGTAGTGTGGTTAGTTGGCTCCTATCTCGCCCCCGGACGGCATGACACTCCAATTAGATACACAAGGTTACAGCCCACTCAAGTGAATTGAGGTTTGATCAGAGATGGAATACGAGCAATTCTTACGTAGGCGATAGGTGCTCGGGTGGGCGTGAAATATCTGTAGTACATCTGGCTGCATCTGCTCAACGTGCCTTGAGTGTGCTCGTGATCCCAGGCCACGCGTTCTATACCACACTATCAAATGGGGTGCTCAAACATAATTTGCACTTGGTCGCATCCGTGAGAGCAGACATCTGACGGGAGGGTGACCTCAGCTTGAATATGCTTGTTACGGACCCGGTTGTATGCTTAGCTAGATGAAATCCAATGTTGACGACTTGGTCAGAATTCGTATATTCATTCCTTTACCCAGGTTAATGGGCTCGACGGTAAAGAAAACGATACAAGGGTCACATCCCGGGGTTAAATTAGGGCTACCACGATGCCTGGCTGTATTCCGGGGTAAATGAAATGTACCTAGTGTTGTATTTCGGTAATATTTGTGAGGTAATACTCTGGGTCATCCAAGGGTTGCTTCAGGGGGCCCACTTACCACCTTCCAGAGGACTACTCCCTGCGGCTTTTGATGCAACGGGGATTTATAGCTAGGGCGGACGTGAGTAGTACGTCAGTATACTCGCGGTATACTGCGCGCTCCCACTTGAATTAGTCGGATCAGAGTATGCAGGCCTAATGCCCATAAAACGCAAGGCCGCTTCCTAGGGCCAAACGGACCTAGGGTAGAGGATCGAGAGAGCAGCTCTGACATTCGGCGATGCTTGTGGGATATCCAAATGAACGTTCTGAATTTCGACTTTTGATGCGGGCTGCACTGCCGTAGGGGGTAGGTCCTTTAGCCATCGGTAACAGGGACTTGTAAACACAAGGAAGATCATCGCTAGCCGGCTCCATTGTAACATAGGTAGGGACGTTTGGCCTCCATCTCAGCATATTAGAAAGTACAAGTCTATACTTAGCCTCTAAACTGCGCTTCATATCTAACTAAATAAATATAAAGCTTATTGGCAAGCTGATATCTCCACTTGATACTATTCCTAATAGTGCACGAACCTCCGGTTTTGGATGCCTCTACCCATGCGACCCTTCCTATCAGGCGTGACATAATCGGACTTAAAACACACATTGGGTGGAGAAGCTTGTGTACTGTGATTCAATCAAAGACCTAACCGACCCAAAAGGCGCCACAGGACAGAAGGATTACCTTAGCCGGGGCAAAAATTACACTTGAATGGAAGGCAGGATGCGAGTGCTTGGATTTCTCCACCTATAGCAAGCCCGTTCAGACTTTCCCATATCACGGCCATGGACTCGCCCATAACACAAACGAGACGTCAGCGACGTGCGTGTATCCGAAACGCGTTGGAAACCCTATACTAATTACTATGTGTCTTCCCGATCGTTTTTTATATACAACAGATCTTAGGTTGGTGAGAGGTTCGTAGCTGCCATCGGCGCCTCAAGGAAGGTGTGTTCGCTATATCATCCACGAAGGTGGACTTATCCTTTAGGCAGGCATAAGATATGTCAAGTCGTCCCTGCCATGCAGTTAAGGACGTCGAAAGCGACAAGTTTTGCGATAGAGACCTATACCTCTGTTTGGCGCTAACTCACTCTGGAATCATCGTACGTTGGATACGGCTCTGCATCGTCAGGGTACGAGGGTCTTTGTTCTCGTACGTCAGGTTGTTGCCGTAGCTATCTCGTACCTTTGTAGACTCGAGTACCATAATAATAACGAAGCTATTGGTGTCGATCGGCGGAGCATAACGTCATGTGTTTCTCGCGCGTCACCCGAGCCCTCATATAAAGTGAATGTGGACACCTTGCTGCCCCGGGGATCTCGTATGAATCACCGAGTAACAGTAGATAATATCAAGATGTAGGCCCGGGAGCATGCGGTTTGCCTAGCTGTCCCCATTAGCTTCGCTGGACTTGTCGGGCATGTAGTCACGTTCTCTCAGTTAGTAGCAAAAGATTAACTCTGAAGTATGGATACTCTTATGCCCTCCGTGCGTAGACCGGATACGGGACTCTTGGGTTTAAGCTATATATGGCGTGAGTCCTAGGCGAGGGTTCAGACTTTAAAGGGAAATGGCAGGAACAACTGGTACGCTAGCCCATGTGGTAGGAAGCAACACTTAAGACTGACGTCAGCGGCGTCAGGAGATGAGGACCAGCAAGGTACAACCTACCACGATGTCTTCGAAGCAGCTGTCTACGATAACCGAATAGGGCATGTACTCACATTGAAGGGTTGCACCTTATCTTGCGTCCGCAGCGCACTATTTCGTCGCCCACAACTCTATTGTTTTTCATGGTCAGTAGCGGGGTACTTACCTTCCTTAGCAATGACCGCGCCGTTTCAGGTCTATCTGCAGGTTACCAGTCTTAGGGAAGAGCCGGTGTAGGTCCCTAATGTCAAAGGATGCGATTAGACTGCATATTTAGGAGCAGTTAGGCCAATGGGTAGCTCTAGTCCCCTCGGTCTTCAAAATGTGCAACTTGGGCAAGGTGGGCCAGAATTGGTTCCCAATAGACGTACGCCCATTCTGAACGCTAACGCTTTGAGGGTTGTCTGCGATGCTTCGGTATAACAAACGAATTGATCACACGCACCTGGACGGGTCGGCCTGAGGGATTTTGAAAGGCCTGTGGCCTGCTTAACTGGTTGCCCGCGAACATCTCGGGATACTCATGAGCTGAATGTAGACTGATTGCATATCGTGCCCTCTATTGAAGATGAGCGGCTGATAGAGCGTAGAAATGTCGCGGTGACCGGTCGTCAATTAAGCGGCTGAGCACTTATGAGACCGACATTCACCTGTGCACGTAATAGGAAAGTAGGAAGGGACCCAGAGGGTTAGTCGCGCACCCAAAGTCTCCTAGTATGCCAATAGCAAATTAGCGCCTCACGGTTGAATAGTTGGGAAAGGAGTCGGCCTCGACTACCGTAAGGCTATCTTACCTAGCCAGGACTTGCGTACTTAATTAGATCAACACCATTAACCGATTCGATACGGGATCACCACTAATCACCTTTGATTTGTACAAGCGGGCCTAAGCCATTGACCATGCGATTAGTGAGCAGACCTCGGGCTCGTTAGATAGGATTTGGGGACTCCAGCCTACCTACGGCGCGTGGACATCGGGCAATTACCGACCGATCCGTCCCATGACAGTTTCTTTCTGTTCTTAGTGGAAATCCCTGGGTTTACCGCGCTGACGAACATATGTCATTAACCTCAAACTCCTCCCCGGAGAAAACAATCCATTAACGCGGAATCTATCGCTGGAATCACTGGTGCCTCAAAGGAGCCACGTATTATGTTGAAGATTCATTTAGCTTTACAAGCACAATGAAAGCGATTATTCGTTCATCCTGTAAATAATTGATACCTATTTTGTGTGAGACTACCAAGGCTGATAGATGCCGTCGCGAATGGGGGGACTCACGAGACAAGTCCTCGGCGGTGCCGCAGTCTCGTCCTTCATATGTCGAGAATCAAACCACTATATAAAGCGCGACTGATGGATCTTCTTGTTCATCTGAGTAGTTACGGTCGCGCCTGGCGGCAGGCCTCCGAATAATGTCCAACATGACTTGCCCTGCAGAAGGTGTGTCTCTGAAATATGAATCCGGGTGTTCGTGTGCATGTTACGGAGGTCTTTGGATCTGTCTACCGTGGTGCTACCCTGGATAGCCCTCCCCAAGAGTCGTATCAGATATGCTAGGGGGATTCACGTTAAAGCATCGTGCCATGGCACATTGTTCCATCGGTCTAAAAAGTGTACCCAGTGCCGGCAAGCGGAAAATAGAAAGTGTCTCCGCAGCCGACACATCAAGACTACTCCACCGGGCTTAAATTCTCTGGTTTGGAACGGAGCAGACATCAATGCTTTCTCAACCACTCGTGGGCGTTGCACGAATGCCGGAGGCCCCGCCTCGGTGTTCGATGGGTGTGGTCCTCCCAATGTTAACACTTGGCACATACCTTAGCATCATCGCGACTATCGAGCGATCGGTAACCCGTATATGACAGTCCTACTGAATTCCTACGCACTGCTGTGAGGGAGTTTAGCAAGGTCGACTACCCCTTGCAGCCAGCCGTTGCGCAGTCCCCAGGCTTTGCCCCTGGATTTAAGTATACCGGTTCCCGTCAGCGCGGCATATGGCAAGGTCAGGTCTCCATACCCCCCTAGGTTGTGCCTCCCTATTCGGCGCCTAATATTAGAGGCATGAAGTTATATACGGGTAGTAGCCGCAGGCTTGTAAGCACGGTGCTTACCCCACCCGGGTTCTTATAGACCCGGATGGACCGGCGCGCGCCGAGAGGATAATATGTTAGATCGATGGGGTCTAACGATTGCCGACCGGTCGGCGGTTGGCAAAGCAAACTTCAATAATACTGTGGCAACCTGAGTGCGAGAAGCCTTCGAACGTTGGAAATAAGCAAATCGTATTGGGTGCTAAGAGTTATGTACGTCGCATCGGGGCTAGTCTTGCTTGCCGCGTGCGAGTTAGCGGACAGCCACGCCCGGTTTTGCAGTTTGTTGCGTCTTAATGCTGGACGCCAAGCGGTAGCGTGATTTTTAATATCCCGGGGGGCTAGCGAGCCGCAGGCTTATTCTTTAATTCCATGACTAAGTGTCATCTTGATTGTCGATCGTCCGATCTCCTATCTTTTCGCTCCGGTATGAAAGACGACTTCGCGCTACGACCTCAAGCCC'
    # mismatches = 4
    # ans = approximatePatternMatching(pattern, genome, mismatches)


    # pattern1 = 'ACTTGTTCACTCTGAGACCTGGCGCCGGTGCGTTGAGAAGGACTCGTGGCCTGCCCGAGAAGCATTTGGAGGGAAGAGGACCCAACTCCTAAAGACCAGCCCCCAAACAGGCCGGCCGCATAAATGCTCACGTTGTGCGCTACACAGATCGAGGCTACACAAACTGGATAATCTAGAACACCCACGGACGGCTGAACCCGGAGACCTACGAAAGGTACCTCGCAAGTTTTCACCAGGTTGCGCCGGCTACATACCACTCGTATTCGAACCGCTCCGTAGCTGACCCTAGGGCCCTGTCGTGCTTTCGTATGGCTAGATGGGGCGGAATTAAATCCAGTACTCCATCGCTATTAGAACATCTTGGTTTATGAGCTACTGAATCCGGCCTCCAGCACATTACTCATTGTAGTGTGCTGATTAGGCGAAGGTATCGATTCATTATCATGTGCGTAGGTCGAAGTTATACGCATCCCAAAACTGCGCCTTGGAGTCACTGCCCGACCTCTAATAGGTGATGCGGAACTAAAGATTCGTTCCTGCCCAGTCCTAATAGTGCTGGTGGAGGACCGAGGTGGTAGGGCAATAATCAGCTGGATGGCACGGTGCCTGGCTACCGGAGCTCGGGCGCAACTGCTCCCAATGCGTGTGCACACTTACTGGGCCTTCCATGCTAGTCGGTGGTTTCGATCTTCAGACGATTGGGTGTCCCGAGTTCATTGTATGATGATCGTACCGAAGGTATGGGCCGAAAAAACGCGACCTATAAGGTATTCCACACTAGATTGTCCCTCGTACATGTAATGCAACTAATAGCGATCCTACTCGCGTGCTTCTATCAGCATAGCAAATATTGCGCGCTCGGTCTGGTTGATAGGCTCGCTAAGATTGCGACGAAAATCGTATACTAGCTTATTTAAGAGGGAGTGAAGACATCTAGTTATCAGTGGGTAGGGTAGACAAGAACTGGAACGTGACGCTGTACTGAGCTAAAGTAAGTTAGCGGAACACATCAGACTTCGACCGCAGTCCGGGCCTGGGGGGGCCCATAAACAGACAAGCACGAGATCACCTTTCTCGA'
    # pattern2 = 'GCCACCGTTAGTACAAAGTGGTATCACATCTATACCGTCCCTAGTCCCCTCCCCTCTAGTCAGTTACGAATTCTAGGATGATCTCTAGATGGTTAATGAATTGATATTGGGGTATGACCCTAGATCCGAGTCCTAGAAAGAGACCGCCGTAGATACAACAACCGAAGACTGAATAAGGAACACAGCTTGGTATAAAGTTGGACATCGGACTCTGTCTTCGCGACTGGGACTATGCGGCCTGACCCGCAGACTAGAGTCATCCACTTGACATGCGTCGTCGATCGGAAGTCAACTGCGGCGCAAATATCAATTAACTTTGGATCGTCATTATGTTTCGAAGGTATTATAGGATCTGGTGGGTCGGCCTCGTGCGCTCGCTCTGGCGGTCTGGAGATACCTTCGAACAACTCATCTGTATGGGCAACCGTAACGCCATATCTGAGTTCGTGTGGTCGTGGCACCAACCGTCTCACCCTGGTTTGACTAACCCGGGTCATTCTGCGAATAATAGCATCTTATTGGCTCCCCTGTGCAAGAGCGCTTAAGCGGGCTGGGGGTGATTCGTCGCGAGTATTTGATGAGCCTTCCTTAGAATTTATTCGCTAACGAGTGCAGAGGCTGCATGGGTTCAACCTGAGCCGGGTCTTGAACGAATTGGGGCATATGCTAATGTGAAGGCGTAGGGTGAGCTGAACGTCTGCTCGACGAATGTACTTACTTGGGCGGGATGAGGCTGTACGGTTTCGGCACCATGACGCCATTTTGTCGTCTACCTTTGTCAAACTCAGGGATGTCTGTCTTCGTGTACACCCGTAAGCGATCGCGGAGTATTCTCCCGGGGTTGGGAACCCTGTTGTGAATAGGCGTTGTGATAGCGGAATAATTTGGCATGAGGACCGGAGATATGACGAGGGAAATAATGCTGCACGGTGTGACCGGGATCCCAGGGCGTTTGCCCACTCTAGTCTAGTGCTTGCGTGAGAAGAAGCTCTAAGGGTAAACTGTACCCTGTCGGGCGATTTAGCCGCCACGAATCTACCGAGGGAGTGTATGAGTTTGGCAGGACTGTCATCTCAGG'
    # ans = hammingDistance(pattern1, pattern2)


    ##### problem 4 #####
    # genome = 'ACGTTGCATGTCACTGCAGGAGTCGGGAGTCGGGACTGCAGGCTAACAGTAGTCGGGACTGCAGGGGTTGTGGGTTGTGACTGCGGCTAACAGTGGTTGTGACTGCAGGAGTCGGGCTAACAGTGGTTGTGGGTTGTGGGTTGTGCTAACAGTACTGCGGACTGCGGACTGCAGGACTGCGGCTAACAGTACTGCAGGGGTTGTGGGTTGTGACTGCAGGCTAACAGTACTGCAGGGGTTGTGCTAACAGTGGTTGTGAGTCGGGAGTCGGGACTGCAGGGGTTGTGACTGCGGAGTCGGGACTGCAGGACTGCGGACTGCAGGAGTCGGGAGTCGGGACTGCGGGGTTGTGGGTTGTGACTGCAGGGGTTGTGACTGCAGGCTAACAGTGGTTGTGGGTTGTGACTGCGGCTAACAGTGGTTGTGACTGCGGACTGCGGGGTTGTGCTAACAGTACTGCAGGCTAACAGTCTAACAGTACTGCAGGACTGCGGAGTCGGGAGTCGGGAGTCGGGGGTTGTGACTGCGGACTGCGGAGTCGGGACTGCGGACTGCAGGGGTTGTGAGTCGGGGGTTGTGCTAACAGTACTGCGGAGTCGGGACTGCGGACTGCGGCTAACAGTAGTCGGGAGTCGGGAGTCGGGCTAACAGTAGTCGGGCTAACAGTCTAACAGTACTGCGGACTGCAGGAGTCGGGACTGCGGAGTCGGGCTAACAGTAGTCGGGAGTCGGGACTGCAGGACTGCAGGGGTTGTGAGTCGGGACTGCGGAGTCGGGAGTCGGGCTAACAGTGGTTGTGAGTCGGGCTAACAGTGCATGATGCATGAGAGCT'
    # k, d = 5, 2
    # ans = frequentWordsWithMismatches(genome, k, d)


    ##### problem 5 #####
    # genome = 'AGGCGAGCCCGAACTGACAGGCGAGCTCAAATATTATCAAATATTATCAAATATTACCGAACTGACTCAAATATTAAGCACTTGCACCGAACTGACAGGCGAGCGTGCGAGGCCCCGAACTGACCCGAACTGACAGGCGAGCAGCACTTGCAGTGCGAGGCCGTGCGAGGCCGTGCGAGGCCCCGAACTGACGTGCGAGGCCTCAAATATTAAGCACTTGCAAGGCGAGCTCAAATATTAAGGCGAGCAGGCGAGCCCGAACTGACAGCACTTGCACCGAACTGACGTGCGAGGCCGTGCGAGGCCAGGCGAGCTCAAATATTACCGAACTGACTCAAATATTAGTGCGAGGCCTCAAATATTAGTGCGAGGCCCCGAACTGACAGGCGAGCCCGAACTGACTCAAATATTAAGGCGAGCAGGCGAGCAGCACTTGCATCAAATATTAAGCACTTGCACCGAACTGACCCGAACTGACAGGCGAGCAGGCGAGCAGCACTTGCAAGGCGAGCAGGCGAGCGTGCGAGGCCAGCACTTGCAAGGCGAGCGTGCGAGGCCAGCACTTGCAGTGCGAGGCCAGCACTTGCAAGCACTTGCAAGCACTTGCACCGAACTGACCCGAACTGACCCGAACTGACTCAAATATTACCGAACTGACTCAAATATTAAGGCGAGCAGCACTTGCAAGCACTTGCAAGCACTTGCACCGAACTGACAGCACTTGCAGTGCGAGGCCTCAAATATTACCGAACTGACGTGCGAGGCCCCGAACTGACTCAAATATTAAGGCGAGCAGCACTTGCACCGAACTGACTCAAATATTAAGCACTTGCAAGGCGAGCGTGCGAGGCCTCAAATATTAAGGCGAGCTCAAATATTAGTGCGAGGCCTCAAATATTAAGCACTTGCA'
    # k, d = 5, 2
    # ans = frequentWordsWithMismatchesAndReverseComplements(genome, k, d)

    # k, d = 5, 2
    # dna = 'TATGGTCGAGAGACCATATCCGCGA TCAGTACGATTTCGGAGGGGGAGCG GCTAGTATGATCAGGAGCCGATAAC TTGGGACTGGTGGAATGCTAGTGTG CTGCAAACGGCACGGTCCGGAGGCG CTATCTCAGGCAACGCGCCTTCGGT CCAAAATGCTCTATCTTCCATACGG GATAATCTAGATTCATAAGGGGCAT GAAGTTAAACTCGGGCAAACTTCGT TAAGGTTGGCCTAGATGTGGGGCAC'.split()
    # ans = motifEnumeration(dna, k, d)


    # genome = 'GCAAGGCGACCGCGTAACATCAGCTGATGTGCACCCCAGTTTACCCGGTCTCGCGTGCCCGGCAGGGAATACGAAGCATCCTCTGCGATTAGTGATACCTTGTGCCTATTTCTTAGGGTCCCCCTATGGAGTCACTAACATACCGACCTCTAGAAGGGTGGGATACACGGCTTCTAGGTAATTGCCTTTACGATGATTTT'
    # k = 7
    # profile = []
    # for i in range(4):
    #     profile.append(list(map(float, input().split())))
    # ans = mostProbableKmer(genome, k, profile)


    ##### problem 6 #####
    # k, t = 12, 25
    # dna = []
    # for i in range(t):
    #     dna.append(str(input()))
    # ans = greedyMotifSearch(dna, k, t)

    ##### problem 7 #####
    # k, t = 12, 25
    # dna = []
    # for i in range(t):
    #     dna.append(str(input()))
    # ans = greedyMotifSearch(dna, k, t, laplace=True)


    ##### problem 8 #####
    # k, t = 15, 20
    # dna = []
    # for i in range(t):
    #     dna.append(str(input()))

    # ans = None
    # for _ in range(1000):
    #     cur_ans = randomizedMotifSearch(dna, k, t)
    #     if ans is None or motifScore(cur_ans) < motifScore(ans):
    #         ans = cur_ans


    ##### problem 9 #####
    # k, t, n = 15, 20, 2000
    # dna = []
    # for i in range(t):
    #     dna.append(str(input()))

    # ans = None
    # for _ in range(20):
    #     cur_ans = gibbsSampler(dna, k, t, n)
    #     if ans is None or motifScore(cur_ans) < motifScore(ans):
    #         ans = cur_ans


    ##### problem 10 #####
    # data = open('../../Downloads/rosalind_ba2h.txt').read().split()
    # pattern = data[0]
    # dna = data[1:]
    # ans = distanceBetweenPatternAndStrings(pattern, dna)


    # path = open('../../Downloads/rosalind_ba3h (1).txt').read().splitlines()
    # ans = pathToGenome(path)


    # k = 50
    # genome = 'CTCATCCGCACAGAAACATCTTGCACAGTACACTATGGCCACAGGAGAAGGCGTGGGTATGGGCCTGTACTTGGGGTGGTAGTGGCTCGAGACAGGAGGGCGGCCGTCAAAGACCAAAGGTAGCTCCAAAGCTGATTAGACGGACGGAACAATGCCGTCTATCTTGCATACTAGCGGACTCAGTAATCCGCACGTTAACACAGGGCTGGATTCATTAATCATACGTTTAGACCCAACAAAAGGCTTCCGTACAATAGAACCGTCAGAAAGGGAAATTTTCTACACCCCCCAGGCCTACTCGAGTGCTGTAGTGGAGGTAGCGCGTCCTTACACAAGGTACTAGTATTTATCTTCCCAAAAGTCGTGTCGATGCCCATTAAGCGCTCCTCATTCAGGCGCGTAGTAAGACCTATTTAGTGTGGGCGACGTCACGCAAACGGGAGATCTTCCACACAAGTGGGAACTATACGGACTACCGATCGAACTGTCTGCTCGCATATCCTTGATCCTACTAAGGCTTATGACGTAGGCTTACTGGAGACGGTAGCCAGTGGATTGACTCAGTCCTACAGCTAATTATCCGCCTAACTGTCAAGAACAAGCGAGAAGGGTCCGCCGCCTGCCTGGTTTCGAATGCTCACGAGGTAGGATCAAGGATTTCAGCGCCGACCACGTGAGCCTAATCGGTTCGGGTTTATAGATTTGTGCACTGGCTAATAAACGTGAGCTTTTGGACCCATCGGAGCTCCTCAAACTGCCCACTAGGCGGCATGAATTGGAAAGTACCGTATGGACACCGATAGCTGGAGCGCCGCTCGCTAAGCCGTGGGGAAAGGGTTCTTCATGCTCAGGCAGCTTATATCTGAAGGACCGTTTATAGCCCCTGGAGGTGTCCCTACGGGCTAATTTAGGCCGGACACGGTCGTTATCATTATCAGAGACAACTAACCGCACCACAAAGACAATTGAAGCGTCATTATCCGACGGTTTTTTCATTGTA'
    # ans = composition(genome, k)
    # with open('ans', 'w') as f:
    #     f.write('\n'.join(ans))


    # patterns = open('../../Downloads/rosalind_ba3e.txt').read().splitlines()
    # ans = deBruijn(patterns)


    # graph = {}
    # with open('../../Downloads/rosalind_ba3g (4).txt') as f:
    #     for row in f.read().splitlines():
    #         v1, v2 = row.split('->')
    #         graph[int(v1)] = list(map(int, v2.split(',')))
    # ans = eulerianCycle(graph)


    # graph = {}
    # with open('../../Downloads/rosalind_ba3g (8).txt') as f:
    #     for row in f.read().splitlines():
    #         v1, v2 = row.split('->')
    #         graph[int(v1)] = list(map(int, v2.split(',')))
    # ans = eulerianPath(graph)


    ##### problem 11 #####
    # data = open('../../Downloads/rosalind_ba3h (1).txt').read().splitlines()
    # k = data[0]
    # patterns = data[1:]
    # ans = stringReconstruction(patterns)


    ##### problem 12 #####
    # k = 8
    # ans = kUniversalCircularString(k)


    ##### problem 13 #####
    # data = open('../../Downloads/rosalind_ba3j (3).txt').read().splitlines()
    # (k, d) = data[0].split()
    # paires = data[1:]
    # ans = stringReconstructionFromReadPairs(int(k), int(d), pairedReads(paires))


    ##### problem 14 #####
    # patterns = open('../../Downloads/rosalind_ba3k (9).txt').read().splitlines()
    # ans = contigGeneration(patterns)


    ##### problem 15 #####
    # data = open('../../Downloads/rosalind_ba3l.txt').read().splitlines()
    # (k, d) = data[0].split()
    # paires = data[1:]
    # ans = gappedGenomePathString(int(k), int(d), pairedReads(paires))


    # rna = 'AUGCAACUGAUGAUUGAGCAAGUAGACACUUGCGAUAGAGUCAUAUCCCGAAUGGCCCUCUGGCCAGGGCGGCCUGCUCUCUCGCACGGCCUGACUGGCCGGCUAACUAUGUUUCUCAGCCAGUGGUUAGGCGCAGUGGGACUUGAGAUUCACCCGGGCAUCGAUUAUGGAAUGAGCAUGGUAAACGCAAAGGCCCCGAUCACAUCGGCUAUUUUUCAUCCACAAGGUAGUACGCCGUAUGGUUCACGGAAAACCUACCUCCCCGGCGACGGUCUCUGGGUGAGGGGACAUCUCCGAGACGCACAGUUGUUACGGAGCCUGAAAACGUUAAGCAUAUACCAUAUGAUAGAUUCUAUGGUGAGCUUUGCCUGUAAAAAAAGGCAGGUAACUCUAAGCGGCGAACUCGAACCGUACAGCGACAGAUUCAUGAGUCGUCAACCAGACGGCGAAACCCGACCUUUGGCAUAUCCAUGUGGUGCGUCACUCUGGGGAAGAAAAUCGGUGUUGCGGCAGACCCUGCCAGUGUCAUACCCAUUGACUGGGCUCGGGUUGGUAUAUAACGUACUCGGUACAACUAUCGAGUACUCUCUCAAUACGAUGUUAGCCUUUUGUCAUACCCGUAAUCACGAACAACUCGUAGGCCAGCACAUCCCCGGAGUUCUCUCACAGGCUGAAGAUAGAUAUACCCUGUCGUUGGGCUCUUUCGAGAAGAGGCUUUCGCUACAUCGCUAUGGCGAGCUACCCACCAAACAGGACAAUUCGCUGUUCUCUUUACGUGACUGCCGCCUAUAUGACGAGAGAACUCUUUGGAAGCAAAUGUAUACGGUGUCAUGUUAUCUGUACAACAAUCCGUUCUACGGCUGCAACUUCCACUUACUAUCCACACUCAUGGGAGAUAGGCUCGGGACCGGAAUGUGUAAUCAUUUCUACGGCGAAUCACGUCGUGAUACUAGUGUGGACCGUUUCCACCACGAACCAUGCCUAAGACGUAGUGUCGGAGCCGUGUAUUGCGGGCGCUUGGAGAAACCGCCCAUUGAAUGGCCUCAACGCCCGCUCCAACACUUCGCCAACAUCAACCCUAUCCGUGGAGUUUCAAUGCUCCGGCCCUGGAGAGUUAGUGCAGAUCGGUCAGCUGAGUCCCCCCUUUGGUGCGAGUACCGACGUUUAGAAUGCGGUCAGCUGACGAGACUUGGAGCGUUCGUCGGUAAGCCAUCGCCACUCCUACGCGCUCGCGGAUGCCUGAAAUCACAUCGCCGAGGAUCAGCGAAGGGGUGUUAUACGCGUUCUGAUAACGGCUCAAGCGCGUCCGUGUGCAGGUGGGCAAACGACGUGUGUAAGUGCACAAGCAUUGCAGAGUCGGAUGCCGAAGCUUGUUGGAGUUGGGCUUUUCUGGAAGACAGUGCUGACAUACGCUUUCUCCGCCGCCUACGGAAGUGCACGGGGGCUAACCGCACCAGCCUGCCCGACCGUGAAAUGCAGGAAUUAAGGGGACGACGAAAAUCUGCACCCUCUUUCGCAUUGCUAGGAAUAUUCCGAUCAAAGUCGGUCGUACAAUUACUCUACGUAGCGCGGAACUUCAGGGAGCUGUCCGUGUGCCGACUACCACGUUCCAACCUUGCGCAAUCUCGGCACUUCCAAGGGAGUGGGGAUGUGGGAAUUAAUCCCUCCAUAUGGGUGGUGGGGCAGCAGCUCGCUGGAUUGGCGAGCUCGCUAGUUAGUAGCCAAGGUCGCCGUCGGCGCCUAAGAUUCGGAUGCCUUCUAUCGAACCUCUGCUGCGUUACUUUUAGUAGGUUUGCCGUGUAUCGUGUUCAAAGGUGCCCCUGGCUGUGUGCUAUCUAUGAAACGCAGAAGAAGCGCCAGUUACGAAUAGGUAUCAGAUGCAGUCGCAAGGGACAAGUGAAUGCGUGCCUCCGAUUGGGGACGAGCAGGUACACAUACUUCUUUCGCAGGAAUGGGCAAGUACUUGCCCAAUUGUCUGUUCGUGACAGGCACUUAGGGAACCAGGCACAGAUAAAAAAGCCAAUUGCGGCUUCAUUAAAAUGCAAUAUAACCCCUAGGGUCAAUAUAAGCUUGUUCGCCGGCAGAGAACAACUCCUAGUUCAUAAAAACUCGACUGCGUUAUCGAACAAGCGCGGAUCCGGUUCGUCUGUAUAUGUACGCAUCCCCUCCGUUAGACUAUUCCGAAAAAGCGAUAGAUGGAUUGAAGGGAAAGCUAUUUGCUGCGCUACAUGUAGCAUUUCGGUGGGAGGUUCAGGUGCCGAAUCCUGCAAAGAGCUACAUCAGCCAAAAUUCUGGCGCAAGGAUCCUACCAGAAGGGAAAACUGUAGACAUUUCGUCUGGAGAUUAGAGUUGUGGGAGGCGUAUAGACAGAAAAUUACAGUUACGCACUCAAUGAUCAUGUUGGCCACCCAGCCUAGGCUGCGCCUAUACGUCAAUCAGGGGCCAUUAGGUGCUCCGACAUAUGACGAAAAUUCGCGGGUCCGGGAUAGGAUGAUCCGCCAUCCGUUGUCGGUGUGUAGGGCUCGCCAGCAUACGCAUUUACUGGGUCGGGCGCUACAGGGAAGAACAGUACCAUUCAGGUACCUGCACUACGCGCGAGUCGGACUGAAGGGGGCGUUGUCCAUCAUCAAUAUAAGCGGGCACAAUUAUCUAGGGCCAGCAGUGCCACGUCAACACAAGAAUCUGUCGCCAAGGGUGCGCAUAGAGAUAACGUAUCAGGCGAAUUUGCAGAAAGUCUUAUAUGACGCUCAUGAAGAAUUAUUUCGUUUUGCAAUGGUCGCGGAGGACUUCGUGGCCUCUGGUUCGUCAGUCGGUCCCACUCUUCUCUUUUUAGUUCACCACGUAGUCACCUGGUGGCCCUGUGAAACCGGUGCAAGCCAACGAGUAGCGACAUUGAUGAAGUUCGGCGUAUACGUAGUUCGGAAGGGGCCUAUUCCAGCUCUUGACUCCACCGUUCUCCGACGACCGCAGGAGAGGUUCAACCGCAUGACCGCCAGGCCUCCUCGCGCAAAGUCAGCCGGUACCCGCGCUGCUGGUGGCAGCACCUCCUUCUACGCGAUACCUCACUCUGGGCAUGUUACUAAUCAGGUCAGUGAUCGAGGACGGUCUUGUAGUGAUAUUGUGAAGUGUGAAGAGUGUGUAUUUCGCCACGACUCGGGGAUUUUUUCCGCCCACUACUUAUCAGAGACCAAAAUGCUUAAUGUGAAACUGCGACGAGAGCGCCAGCAUCCGCUAGUUUCGAAUUGUCAAGAGACAAUAUUCUCAUUGGUUGUGUUGAUGGUCAGAAUGAUGUGUAUAGUUUCAAAACAAAGUCAACUUAGCCAAGAAGAGCCGCUCGCCCGCAACCGACUCAAAAAAACUUCCAAGCCAGUUAGCCCACGGUUAGAAGCAUCGGAUAGUAAGCUAACGAGCCCUUUUCUACUGCAUGACUGGAACGCCUUACCGAUUAGACUAAUGGUACGUUGGCGGACCGCUCAGCUUAAAGCCCGCGGAGGGUGCGUCUUGGGCCCAGCAAUCACGGGCAUAAAUGGAUGGACUAGGGUCUGCGGUGGGGCAAAAGGGGCGUUGGAAGCCAUAGCACAUUACGUAUUACCGCUCCAGGAGCAGGAGCGUAACGAGCGUUCCUGGAGUAGAUGCGAGCUGUUAGUAGCUAUGCCUACCUACUUCGUCUACCUAGGGACCCAAUUAUACACUGAGCCAAAUCACGUGAACUGCGCUCUAAGUUUCGCGACUAAUCAGCUCUCUGAUAGGCAGUUAGAUAGCCGGGUGAUCUGGUCACAGACGGGUCCGGCGUUCGCUUCUGAUGUCCAACUCCCUCGCAAUCUUGUUUUGCGUGAUUACCGAUCGGUACUACUCCAUCAUAAUUUUAAAUACAGUAUAAAGGCCGAUUCCUGCCUGAGGGAUGCAAGUCCCAGCAUCUGGUGCCACGUCAGCGUAUUCGGGUACGCUGCAUUCAUGCACUAUACGUUGACAUGGCGUGUUACUGGUUUUCAACCUAUUAGGAAAGGGUGCAUUCUCAGUGCUUUUGCGGCGUAUGAUGACCGUAUACAAACUCUGGCUAUUGCCUAUUCUAUUUGCUCCCCAUAUAGAGGCUGGGACGACACAGUAUGCGAAGGCCGUACGUACACAUUCGUUGUGAUAGUGCAUGAGGACACAUUCCACUCAGGACUAUCGUUCGCCCAACUAGAUUCAGCCAUAUCUCAUGCAGGAGAUGCCCAUACACUUCUAUCAGUGGUAGGUCCUAGGGAUAAGGCUAUGGAAGAGUGGCACCCUUCCCUAUAUUUGUGGUGUCGUCAGGCUAUCAUAAAGAACGCUGGGUGCAGCAGUCGAAAGCAGUAUCUGGGGUCACCUAAGCGGGGGGCGGCAUGUUCCUAUCCCUAUCCAAAUGUAGUUCUCAAGCCGACACAUACGUUACUCCCACAGUUGGUGCCUGUCAAAUCGAGCAUCGUAACAUCUAUGAAUGGCACUCGACUAAGUCGCCUAUCGUCGCUUGUUCCGUGCUCCAUGCGUAGGACCUACAAGAAUUUAACGGACAGGUAUGGUCUUUACCAGCGCGUGACAGCCCACUGUUACUACGAACAAAAAAAUCACCGAAUAACGCGGCGUGCCCUCAAGAAGCAAGCAGCUAUCUCUCCCAGGCAGCGCGGUUUAUUUUUCCGCCUUCAUAGGCACUCCUGUCAUAUUGCUGCUCCGGCAUGUCAUCCAUGUCGUUAUGCACUAAUGUUCUCAGCCUACCGGUACACUCAGUCAUUUCAAUGUGUGGCUCAAAGCUGCGAUGAAACACAGCGCUACGUGGCCCAGACAUUGGAUCCUAAAGCCUCCUUGUCUCGUGACCAUGUGCGUUUAAGUGGUACCCAGGACUGCCGCCCCCUUCACAAAAUGAGGUGCGGUGAACAGCCUUCGGGGAACGUCUGCGUAACUAAUACGUGGACUGCUACUGCGGCUCGCGUAGUCAAGGACCAUUGGCUAUGCGCGAUGCAUUGGACGUUAAAGCUAGGGGCCGCGUCACAGUACACCCCUCAACAGAUGGCUGACGCAAUGUCCGUUUGGUGCCCACGAGCGGAGAAAAGAAAUCGAACUCUACACAGAUCAACCGACCACUCAGAACAAAUUUUUUUAAAGUCGGCAGGCAGGACAAACGAUGAUGUCGUAUAUUUGCACUCGGGUCAGAUGAGUGUGUCGAGACUAAACUAUCGGACGGCUGACGUCUGCAUAAGGUCGCCUUGGGAUGUUGCCGUGCUUUUCACUCUCAAUCGGAUGCAAGUGAUCCGAGCAUUCAAGGACGCAGAUAGCUCCCAAGAAGGUGGUUUUGAGACAUUGACGCGCAGGCUGGUCUCAAGCAAACCCCCCGGACAUCGCCACGUUUACAGGCGGAGUUCGCUUCGACGUCGACCAGUCCCGUUGUACUUUUGUCCAUAUCAAAUGCUCGAGUUAGCAGAUGGGGGCGAAGUAGCAUCUACCGUUCUGCAAGUGGCUCUGCCGUCCCCGUUAAUUAUCGGAAGCGCCGGGCUGGAAUUUCAUGGCUUGAUCGUGACGGCCUCACUCUGCUGUCCAAGCUCUUCGCUGGGGGCUCCUAGUCGAUACUCAGCUAUGUGUCUAAUACCUACUACCAACCAACCGUACGGAACACACAUCGUCCAAGGAACCCAAGUCAGCCCUGUCCCGUGUGUUCAACUGUGUGUUCGACAAGCUUACUGUUCGAACAUCAAGGCAGGCCACCGAACGUUCUCUUCGUUUGGACACCUUGUGGGGGUAUUCACUCGAUCAUCCCCAAACUUCGUACCAUACGAACGCGCGCGUACUUUCACCGAACAGAGUUGCUGCGACGGCAUCCCGUAUAACGAACUGAACUACCUCAUUUUUGAGGUCCCUCCGAAGCUUAAUAAGAAGUACACUAUAUCAAGUUUGCACCUCACCGGGAAAGCGCAGUCUAAACAGAAACGCACGUCUGGUCGUGGACGAUGUCCAUCCUCACAUCCUGUAAUACUAUGGUGGUGGCCACUCCGGGGAAUAGUGCUCGAAAGACGAUCGCGGACCGAGACACUACCAUACCUCUCCUUCACCCAGCCAGUAGUCAGAAUACGCGGAUACCUUACUGGGGUGCAUUCAGAGUCACCUAAUGCACCUCCGUUCCGCGAAAACCGGCGGGCCUUUAUGGCCCACUCACAGCUUGGGUCGGACACCGAUGCUUGGCCGAUACAUGCCUCCUCCGCCCUAGUAGACCUACGAAAAUUAAUGGUCUGGUCGCACCGCACCGUCACUGUAAUCUUCAAUCAGAAAUGCCGCUACCUAUGUUUCACAGCCGCAGCGUCCAAUUACACGGCCCCUCAAAUCAAUGCCACAGCUGCGACCAUUGACGCCUACCGUGGGGCGCCUUCCUACAUGUGCGCCCGCAUUACUUCCAAACGGAAGCCUUGGAUUUACCAGGCACCCCGAUUGACCUGGUUAAUCACAACAGGGUACGAUGCUACUUUCAUUGAACUGAUUCACUCCCGAAUCGACAAACACCAGGAGAAUUUGAGAACCCACCUGCUGGAUAUCUCAUUUGUCUGUUCAGAUCCAUCAGAUUAUGCGGCUACAUUGCGGAUUAGGUUACGAUACAUAUGCGAUCUGCGAGAAAGAACUUGGCGGAGAUGUGCGAGACUCCGCCCGACUCCGUAUGUACAGGGAGCGAUAGGGGAACCGCACUGGGUUCCCACAGUCUUGGUAGGCGAAACGACUACGGUCGACUCGUGCGCCAUCCCAAUUACUAUCCUGUCCGGGUGUAAGUGCAAACGCGACAAGCAUUGGCCAUCGUAUGGUGGUUCCUCUUUAGGUAGUAGGUUUCGGGUUUCUCUUAUAUCAUCUUCAUUCACCCUCCGUAUCAUGCAAGUCAAGCAAUGGAGGCUCAGUAGAGUGCUCCCUCUCGACUUGGACCGCUCUCCCGGAACUGGUGUUUUCAUCAAGAUAACGUCGCAAAGUAUCAGUCCCGCCCCCGACCAUACCCUGGGGGGUGCUACGGGCGGUGCUUUUCCACGGACAUUGCGCUCGUCUGGCCUAGCCAUUAACACACGUCUGCAUUUUCUAGGGAAAAGCGGCUUUGUCCUGACAAGCGGUACAGAGACACGCACCGGAUUCACGGUACACCUUUCCCUCACACGUCCUCCUGGAGCGCUUCGGGCGCCCAUGAGCUUGCCAGCAUUGCAAUUCAACUCUGUCCAACUGUUUCCCUUGGACCUAGUAUCAGAGCCGCCAGGACGUACCUCAUCCGAGGAUCAGAAUGCCGCGUCGAUGGGAGAAAGUCACAAGAAGGUUCCCGAGAUAGUCGACGAGUAUAGUUGCGGCCACUGUUCGCGGAGCAUGUGGAUGUCCCUCCAUUAUAGUAUAUUGCAUAGGUUUCGGGAUUCUCUCACGCGUCCCCACCCGCUAGAGAAUAUUGGCCAGACUGAACCGCCUCUAGGAAGGAGAGGUUAUAUUUCAAAAAACGGAUACCGAACUAGGUCAAUAGCAGGAGACCAUACUACCGUUCACAUAGUUACCCAAAACCAGUUUCGCCCCAGUUUCGACGAACUUCUGUGGUUGAGUUCCCGUGGGGACUGCAACGCAUUGCGUAGCUGCGGUUCGACCGCGCCAGAAACAGUCUGGACCCGUCGAACAACGAUUUCUAGCAAGACGAUGUGCGCAUACCGUGAUAGCCGAACACUCUCAUUGAGGACCUGGCAUCCAUAUCUUUCUGUUGCGACGGGGCAUAGAUCACCAUGUCCCUGCCCGAAAGAUUGCUGUGGAGGCGUGAGCCUUCUACAAACUUUUUAUUCUCAUUUUCGAAGCUGUCUAAGCCCGUCGGGUGAAGAGGUGGAUCUCUGGUUACGGAAGCAUGAAAAGACGGCUAUGAAACGACCUCGCACGUCGUAUGAGUAUUUGCGCCUGUCCGAAGGUCCUGUUGCGCCUGAGCGCCCAACUGAGCUUCCGACACUGAUACGCUGCACUCGGUUGCGUAUCCCGUACUGUUCACUACGGGUAUCGAGGCGACGUUUUCAAAUUGCUUUUUGUGAGGUAGGAUUGAUGUUCGCUACUUCCGAGCUUGUUGGUAUCGGACCUUUCGUGAUUUUCAAACCAUUCGUGUCACAAUAUCAUGUAUAUAACGAACUUAGUGAUACACGUGCACACCGUCUCUUGGGGAUGAGCAAAUAUGUAGAGUCUAGAGUGAGACUCGCUAAUGAUGGAUCUUCACACGUUUCCCAGCAAAAACGAUCGGUGAAUUUUAGAGGGAAUUUGCCACAACUAGCAUUUCAGCAAGACCGCCUUUCUCCCUGCAUUAGCUAUCCCUUGUUGAGGCCUGGUUUGAGAACGGCUGAAGACAUACAUAUCUUCUUUCAAGUUAAUGAGUUCGCGAACUGUCCUCCACCGACGCUAUACUGUGAUUUGGCGCAAUUGCAGAUCCGCUUGCUCCACGACUGGGUUUGGUUGCCAGUUUUGUCGAGUUCCAUUAAACGCGUUGAAGGCACUCUAAGGUGCCAGGGGCUAAGGUGCGAUGCUCUCCCGAACGCAUUUAAUGCGAAAGCGGGCAAUCUGUCAGCAAAGUGUUUCGCAAACUGGGCAGGGAACCAUCAGUAUAAUUCAGUGGAAAAUUUAUGGCGUGGGCUUACUUUGCCAACGUCUUCGUAUUUAGAUUCGCGGAGCUUGGGGAGUACUACAUACCUGGUCUGGGUAAGUUCGGAUCUGCGAAUUUGUGGCAGGUUGCCAUGCCGGUCCAUCGGAGGUUCAACUCACCUCGCGAGGCUAGGAGCAGCGUGCACGGCUUGUUGGGAUCAUCUAGUCUCGAGUGUUGCGGCAUACUGGCUCGACGGCAAACGUUAG'
    # ans = RNA2AA(rna)


    ##### problem 16 #####
    # genome = 'GGTTATAAAACGGCGGGTGGAAGTGGGGAGTTCGCATAAAACCCAGGAGCGGAACGTAGATGGAAATGACTCTCGTGGCGATAGTAGGTCCCGTACCTTGTAGAGCGCTGTAGTTACATGCAGACCGTGTAGCTGGTTGCGGGGCGTCGTTGGATGGATTAATCAGCCACACCGGCGCGCCGTGGGCTGAGAGTGTAGTGAGCATTGTTCAGGTTAACGATCCGAAGGAAACGGGGAGATGAGGTATAAGTACCGGTCATGCGTCTAGAGCTCGATATGGAGCCTTCCTGTATGATAATCACAAATGCCAGAATGCTGTACCGCTGCGCGATCTTGTGGGTCGGACACGCTTGTTCCACGCGGCGGGCTGTACCTAAACTGCGGCCCCATATTGCCGTTGTCTTTCCGAGTCAGCTATCGTGAATATACAACAGCAACATCCTCAGTTGGGTAGGGGGATGTTGTCAGTCAATTGCTTGGTACGGTCGACCTAGTGCGGATAAAAACCCTCTTATTCCCGCTCGCCATGCAGGCACCTTATTGTCTATGACGTTAACTGTCATCAGCGAGGCCTTACGGTAAGTACCATCGAACTAAGGATTCCGAAATTTGTAACCCATCGTTTGGTCGGTTGAATTAGAAACATTCGTAGAATAATGAAAGTCTCGCGAGTGCAGCGTCTTGTAGTTTTTTCCGCTATAAAATTCCACCGGCGATCATAGGGAATGGTGAGAGCGCCTCCTACACGCCATCGCACCTTCATTGCCTGCATAATTAGGGATAAAGACGTAAGCAGCAGGATGTACCAACGAAGGATCGCGGAAACCCTTTCCTCCGAGGACCAAAATCTAGATACTAAAATGTCTTCTGGGAGACTTTCATCATACTTAGGATGACGCTGAGATAACATGTAAAAGGGACTAGATAATTATAACAAACGTTTTCCAACTCGAGACCTTCATAATCTTGAGGATGGTGAGTCGCACAGGAAAGCGCCGTCACTAACTGCTCTAAAGGAGCCATCAAGATGAGCGCTTAATATTTAGGAGGTATCCGCAGCACGCCCATGCGCGGTTTCAACGTCCGATGACTGCAGCAATCTTGTCTTCACCCGGTATTCGCGCGGATGTGTAATCGAGCAAAACAGCACGAACGTACCGTAGGTCTCTTAATCCGGGCGACCCTGTACGTTAGATGTACCACTCGGACACGCCAGATGCGACGAAAGATCGGGAAGTGAATTGCTTACAGCAACCCGATTTCCAATTACTGAGGGGAGACTTTTATTATACTAAGAATGTTAAAGCAGTCTGAGCTTTACGATTCGGCGAGAGCTCCGGTTTCAGAATGAAGCTCTAACATCCTAACACGTGCTGTAGCTACGCACGATCTAATTATCTCTTTTAGGTTTTGCGCCTGCTCACTAGCCATTATACTTTAAGGCATTTTTTGCCCAAGCAGAGAGTTCTTATCTAACCCGGGAGACCATTCCATCAGAATAAATAGAATAAACAGTAAGCATCCAAGCGATCTCGTGACAGAAGGATGCCCAAGTAAATAACCTTATCCCTATGCCAGCACTTTCTGTCACGTGGGACCTTGAGGATCTATATATGAGCATACGATCCTCTAGTGGGGGATTCAGCAGCCTACGACGATCCGCCTTCCTGTGCTGGCTCGGTGGCGGCCTGAGACCTTTATCATATTACGGATGCATTGAATGTGTATGCGTTGCCTAGAATACGCGGGACACAGTAACGAGCCCCAACAAAACAGATTGAACACACCGGATGGAGTCTGTGGTCGCATTCTACGATATTCCAGCCGTCTTGCTTGCTGTTACATACGTAAAGTTGAAAGATGGACACTGAACCGATGATATCATTCGTTAGTCTTGCCTAATAGCATGTCAACGGGTGACGAAGCTAACAAGCAAAGACATCATAGCGTCTCATACTTGCCTACCCTCATTGGCGATCTGCCTGCTTATGCAAACTCTCTGCCGCTGTGTCGAAAGCGCATCCGGAGGATGATAAACGTTTCTGCTGACGCCCGTTGTCCCCTTACCTTCGCATTGGCACGCTCGCGGGAGTCTTACGGCATAATTCGGCATAGACTGGCGGCAACCTTTGTGACCGAAGGTAGGTCTGTGAGACGTTCACCGCTCGCGCATAACCCACTACGTCACCGGGTCTAAGGACATGACTGACTATGGAGCTGTGAGAAGGGAAACTTTCATTATACTTCGCATGATGGAAACCGGCAGGAAATGCGGGACGATCTCTAGGGCGTTGTGAATTTTATCCCTGACTTCAACCGCATACTGCGGTTCAAGGTCAGCATGGGAATCTGTCTAAATGCTTGTGCTCTCCACAGCGGCTCAGTCGTAACAAGTTCCGCATGAAGCTGATTGCGATCCACATCCCTGCGCCTTCCGCCATGTGACTAGTTGCGCGTGCGCTTAGTCCCGCCGTAGTTTCTTGCGAGAGCGCGCCTGGTGTACCGAGAGATGGGACCACACTGGACTGCTGGCCTTCAAAACTGTGCCATCCGAAATACAGCTGTTCAAATACAAGGTCAGAGGGCTTGGACGCATAATGTTCGGCACAAGGTAGGTCAGTAACTTTTGACTTCTGCACGGATTTAAGTAAGTGGCAAGGTACTTAAACTACCAACGTCTACCTAGCGCGCAGATGTTACTTTTGGGTTATGAAGAGCCCTTGCGCAAACAGCACTAAGCCACGCTGTGACGAGGGTCTAAACACCAGGCATCTGGTATATACATTGAGACGTTTATCATCCTGCGAATGTACGTACACCTGGGTTCAATTCCAAAACCAAGTACCTGCTTAGCACTGGACGGCACGTATCGACGCTTCGCAACTATTAGCGCGCGATGGCATTGATAAAAGGGAAACGTGTATGGGCCTATTTGCTCCTGCATTCCTTGTTTAGCCATTAATGTCTAAAGTGCTCTGACTAGTGTCACAGGTCAAGGTGAGTGGCAATGTAGTGAGGAGATGACTACATCGGGTCGACAAGTATGCGGTTCTGCAGAGGTTTTTGGAATGGTGTTGCCATATTTTGCCCAATGAGTGAGATCCCACATGAACCTGCACCGTTCGACCTTGCTCCGGAAAATTAGCTGTAAAAAATTTGTTAACGAGGGATGCGACACCGCAGTCAGTGGAACAGGATGTACCTCACTCCAGCATGCATCTGAAGCAAAGGGCTGACAGATGCATCGGAGCGTTATATAATGACTGTGAAGAGTTTTTCTGTCCTGTCTTACCGGCCCTAAAAGAGAATCGAGTATCTCATCCCTGTAGGCCTCGTTACTCACACCTTCGGTCACTACGATGGATAAGGCTTTCGACGTCCAAAAGCATTGTTAGTGTAGAGATAGGCTGCCGGTTTCGGCCGGGTTAATAGCAGGGCATATAGGGAGCCAGGTGCTTGGCTTCTTCCAACCGGGTATTCTTTAGCTGGGCGGCTTCAGAGTTGGTTTTCCGAAAAAGTTCTTACACGCCCATGACCTAACCCAGGCGCGTTTGGGCGGCATTATCGTCGATTTCCGTATACCATGGAAACATTCATTATCTTAAGAATGGATTACAGGAGATCCTGTTGGTAAAGTTATCATCCCAGATGGTTTTTCACTCGAATGCGCCCCGTAAGGTCGATGGCTCGAGATAATACTCATTTGTCTAATCGGCAGATACAGACAGTTGGCCATCTTGAAGCGTGATCCCGTTCGACATCTTTGTCGGGAATCCAAAGTACCCGTAAACGAAGTAATGACAGGGACGAGTAAGGAGAAACGTTTATTATTCTTAGGATGCCGCTAACACAACGTGTTATCCGACTCGTTTAAGAAACCTTCTTCCTTAGAGCAATGACTTAGAGTACTTAAAAAGATACTAATCTGTGGTTCTCTATGCAATGTTCGATCATTGATGTTTTAGGAGGAATTCTGCCTATATGTCTTCACGTTCCGTCGTTAATCTTTTGTACCGATGTACTTCGCGACTACAGGAGTGTTAGTGTTCGCAGGACATGATATCCCCGAAGCAATAAAAACACCGGGGGGGCAAGTTTGTTTGAACATTCGAGTTACATAGGCACGGTCTGATGGGAGCCTATAACCAGGATTTGAGGGCCCTCTAGGGGTCGGAGGTTACGGTGATCACTCAACTCTAAATCAGGAGACGCCTGTCCCAAGGGACAGCAGAATAATAAGGTTACCGAGCGTCAGACGGTCTAGGGTCACAAGGTATGGCGACTCCGTCTGGGATCTAAAACGTACCCTCCGTGCCAATAGTGTCGAGCACCCGTGTACTAATAAAGTTGACACGCGTATCATGAATTCTACGATGCGAGGTCGTAAATTACCTTCTAAGTAATACGGGTGTTTTTTTATCAGTGCGTTTTCGATATTCGAAAAATGCCTGTGCCATATGTGGATAGACGCACAGGCACCATCGTCTTGGTTCGCATGCCTATCATTATGTCAGAACATTGTGCGAAAAATCAGCGAAAAGACGATCGGCCAGCCATGCGCCCTAACCCGCGTCACACAGCCCCAGAAGACAAATAACAGAGGGACGCGCGTGCGAGCATGCGCAAAATAATAAACGTCTCGTTCCACGTACATACGAAGAATGATGAACGTTTCAATAGATGCCCGGATTCACGAGCAGCATTTACTGGTACAAACTACACGACCGACCTGAGCAATCCCGCCTTAACCACATACGTTCTCGCGCTAGACTACTGGTGGAGAATTCTACTATACGTCGAATGCTGCCTTGACTTCCCAAGCACGGTACATATTTAGTAACAAGTTATAGTGAGCTCCGTAAGAATGCCCCCAAATGCCGTTAAGAGGCCTGTGCATTTTCCGCAGTGTTCGGAACCTACCAATCTTCTCCCGGGGCGTGTACGTAACATCTCTGGGGTGCCGTATGACCATTTTGAGGGTCCAGGACCCGGATGTATCTATACTTGTAAGGGACTACCTTAAACACGCAGGATGCCGATTCCATCGCCCAGCCATAATGCAGATTCGGAGTTGCAATGGTTGTGGGCCCCATATCTCTTTGTACGAGCTACGAGCAGCCGTTTAAGATTTGTGCCCATCGACGCGAACGGAAGACGCGCCATTCGCAACCTAGCTCCGAACCATGTAAACCAAAACCAGACACTACGTGCGTTCCATTCCTTATTCGTTTCACTGGAGGTACGACGTTGTTCTCAATTGGGTGGGTGATCACTACATCCTTGGCCCGAGGCACGTATTTAAGTTCAATCCTCTCCTCCCATTCGGCGAGTTATATCCCATCCGGAGTATTATGAATGTTTCCGAAACAAGATCGCCTGGGTACCCTCATTGCACCCGGCGGCGTAGTTTGTAGCTGAGGTAACAAATGCCACCGTCATTGAAGTGGACGGGGCGGTAATTCCAGATGTGAAAGACGCAAAAAAAACAGATAGCATAGTGGTTGCGTTATCCTTATAGAAGACTCCGCACTTCCGGTCGGTATGGTTACTGTACAATTCTTAACTCGATCACCGTTGTCACCGTATAACCTGTCGTGTAGAGTTGGCACGTTCCGGGTGATAGAACGAGACAGACCTTCGGGTAATTGTGCAGCACTTGGCGGTGCCTTAGACGTACTCAATAGAACAGCATATTACATGTGACATATGTTATTTCCGATCATCGTGTCGAACGGCCGATTCTACCCACGCCGAAAGTCATGCAGAGTCAGATCCGCTAAAGTGTGGCGGTCACACCACCATTTGACAGGACCAAATCATATTTACCCGTGGCAAGTCTGACGACTTCGACAGACGCTTGTTAGAGGCTTTTCTCCCAAAAATGGCCGTAGCTGAGCGCACAAGGATGCTGCGTCTGGTACTTAGTATTTGAATCGACAGAATTCGGCAGAACCGCGTTCGGGCGTAAATCCTCGGACGCGAGTAGTAGACCACGACAGACTAAGGACTTGGCGCGGCTATTGAATACAGATTCTATTGCGCCCCCTCCATTCGTAGGATAATAAACCATCCTGAGAATAATAAAGGTTTCGGGCATGCTGGGTAGAATGCCTGTTATATGATGCCCCTCGCCAGGTCCTGCTTTGCCTCTAAGAACCACAGTAATGCGACAACCGCTATAAAGAGTAGGGTCTGGTATAGCTACGTGTCTGGGCCGCACATGGTATAGCCCGAGGTCCCTTTGTCGCGCGGGGTATACCCAAGTTATAGCCAATCTTACGCTTAGATTATCAAGAGCTGGTTTCGGTAACGTTCAAAAACTTACTGCTCACCGCACTCTTGCGTTGTTGAGAAGCTCTAATATTGGTCATCGCTGTTCTAACCTGATCACAGCAGGGTTCCATCCTCAGTATTATAAAAGTTTCCCAGGGCCGCAACTCCCTACATCATTACCACCAGCATGGAGTCGTTGTTATTGACAGGGAGCACTTCGCCCGTCTTACCGTTATGCGATTGCAACTTCGACCGGTTCTACATTAGTCATGTAGATTATTAGAAACCCATTTCATTCGTCTACGTTGGGCAAGACAGGAATCCACGGACACGACTAGCTGATCCCCAAGGACTTCTTTTTTTATGATTCTTCTAGGCAATGAAGCATCGCTGCTACTACACGCGCCCTAACCGGTATGATGATCCACGGGACCGTATGATGCCAGTTTTCGCGCCCTACAACTACGAGCCTGTGTACGATGGTCGACCGTGGCCTCCAGCGGTAAGATCTACTATGGCGCGCATGTCCGAGTTAGGTGGGCCTGTGATTGTCGGGGGATACGGGGATTTTATGAAAATGTCAGACGGGCCTAGCTCATGAATCGCCGTGGCTCACGGTATCCCTGCCCGCACCTTGTGAAGTGTTAGGGCGGTCGCTTCCATATCCGAAGAAGGCCCAGCCCCGTAGCCGTACTCTCACGTTCAAGTCCTATCACGTAACCCGACTACCCTTGGAAGCCTAGCTTAGTCCATCCGTCCCACGAATAGTCCACACCCTAGTAACGATGGGAAAACGCTGCTTGTTAACGGTTAACGTGGGATCATCATCCAGGACCTTTATGGTCAATCTCCTAGTGCTAACCCAATCTGAGCAGTAAGGAAAGTTGTCCCGCGATGTACCTCGAGAGTCGAAACAGTGTTATGGTTTGTCTCTCTGTTTAAGCAGCCAAGTGAGCTTGCCCATCCAGGTAACCTACCGTTTGCTGTCCGTATTGTATAAGTGTGAATGAGAGAAAGAGTACATAGTACATTAGGCAGATACAGAACCCTCCGCGCGCCGACCAGAAACTTTTGCGTGTAGAAGGAGTGACAGCATCCGTAGTATTATGAAGGTCTCGTGCAGTTTGGGCCCTGCGAAAGGTAAAACAGTCATGGGATGATTCGTTTGCATGGGCCCCAATACGGATAGTAACCGCTAGCAACGGATGTTGGAAGTGTGAGTTATATGTATTCTAAGACCCCAAAATGCTTCCAAATCTTATAGACTTCACGAACGCAACTGACGTTCGTGTTTGCTCTACGTGAGAGGAGGTAAAACGGATGTAAGATAACATCACTGGAGAACAAGGTTAAATCAAGCGTACTGTCGCTGGGAACTTGCAAGTCGAAACAGTCTAGGGACTGAGAAAACCACGGTGCCGGACGTAATTCCGCGGTGCAAGAATTACTATTACATCTGTACGCTCTTTCTGTAACGTCGTGAGATTAGGTCTCCTGTCAGTGAACCCGCCGACTATCTGGACGGAGACCTATCAGTCTCCATACGAACTTCAAGTAACTATCCTATTTCTGCCTCGGCTCGCGGCATGTTTTAGACTCGCCACGGAACTCGGCGTGTCTCCTTAGTCAAATCACTTTGCAGATGTACCGTGTGAAAGCGGGTACCCGACCTAAAACAACGAATTGGACAGTCGAACCCTTCCGGACGAGTCCCACTGAGGGATACGTACTTAGCATTAACGAACGAGTTCCGCTCGCCCTGCAGAGACGTTCATTATACTACGTATGCCAAGCGGCTCAATATACACGTATTGGTACTCCATTGATAATGACCGATTTGTAAAGGGTTAACACATGTAAGAGTGGGATCCACAACATAACATGCACTTTCGTTGGGAGGGTCGGAACGCCCCTTCTGCTCCCTATATAACCTGATACATGGGTATTCTGAATCTTCTTGCCAAGTGAGTCACACGATCTGGCAAGATTTCCCATGTCATTTATAAAGACTCCCGGACCAGATGTATCCGAGGCATTACTGCGGTGGAGCATATCTTTCCTTGCACGCAGGAAGAAGATTACGGAACGTTTGAACTAAGTTGTTTTCGCTTGTCAGCCGTACTCGCAGCGGCTAGTTTTCATCATATTCTCTGTACCGGGACGCAGAAACCTTTATTATTTTACGTATGGGAGCGAGCCTTAGTTCGCATCACGCCTTGCAGCGTCGTATCCGATAAGTATGGCTACATTAGACACCCAACGGAGCACTGGGTTGACCATCAAACGGCACCCCACGTTTTCTCCGGAGTCGGCGCTATGCATGGTTGAAACGTTCATAATATTGAGAATGCAAGTAAACATCTATACTCAATTAATAGATGGAGTCTCATAGAATAGGGTATAGGCGCCGAATATGCTATGTTTTCCGATAACCGCCCAGAGAGTGCCGATCGCAACTGCGGAGGCTCGATTCACTGACTCGGAATCGAAGCGTTCGTCTGCACTTGACCGCCTAATTCGTTCCCAGAACGCGACAAAAGCGAAGTCAGAAGTCCCTCTGCGACGCCCGCAAACGATTATGATAACTTCTAGTGAGGCAGAGGATCACCTGTGAT'
    # peptide = 'ETFIILRM'
    # ans = peptideEncoding(genome, peptide)


    ##### problem 17 #####
    # mass = 1351
    # ans = countPeptides(mass)


    ##### problem 18 #####
    # spectrum = list(map(int, '0 87 115 128 128 128 129 129 147 163 216 243 244 250 256 257 275 276 291 331 371 372 378 379 385 404 404 438 459 494 500 500 507 525 532 532 567 587 622 622 629 647 654 654 660 695 716 750 750 769 775 776 782 783 823 863 878 879 897 898 904 910 911 938 991 1007 1025 1025 1026 1026 1026 1039 1067 1154'.split()))
    # ans = cyclopeptideSequencing(spectrum)


    ##### problem 19 #####
    # n = 379
    # spectrum = list(map(int, '0 71 101 103 113 129 131 137 137 137 147 156 163 186 200 208 238 250 250 268 269 284 285 287 289 294 300 337 355 356 381 387 390 398 401 406 424 431 431 436 458 469 484 493 527 532 535 537 537 544 568 573 587 587 606 606 640 644 666 669 674 674 681 690 700 718 737 743 743 745 753 773 782 811 821 829 837 837 855 856 874 874 882 890 900 929 938 958 966 968 968 974 993 1011 1021 1030 1037 1037 1042 1045 1067 1071 1105 1124 1124 1138 1143 1167 1174 1174 1176 1179 1184 1218 1227 1242 1253 1275 1280 1280 1287 1305 1310 1313 1321 1324 1330 1355 1356 1374 1411 1417 1422 1424 1426 1427 1442 1443 1461 1461 1473 1503 1511 1525 1548 1555 1564 1574 1574 1574 1580 1582 1598 1608 1610 1640 1711'.split()))
    # ans = leaderboardCyclopeptideSequencing(spectrum, n)


    ##### problem 20 #####
    # m, n = 16, 400
    # spectrum = list(map(int, '0 87 99 99 103 113 113 115 128 128 129 131 137 163 163 198 202 227 231 232 236 242 244 244 250 250 278 291 291 326 335 345 349 357 360 365 365 373 378 381 390 394 406 448 463 473 476 480 481 486 489 493 493 494 505 523 528 576 579 589 592 593 596 604 604 610 623 626 636 656 656 691 692 707 717 722 723 725 726 739 741 755 759 767 784 820 821 825 828 838 854 854 854 854 870 880 883 887 888 924 941 949 953 967 969 982 983 985 986 991 1001 1016 1017 1052 1052 1072 1082 1085 1098 1104 1104 1112 1115 1116 1119 1129 1132 1180 1185 1203 1214 1215 1215 1219 1222 1227 1228 1232 1235 1245 1260 1302 1314 1318 1327 1330 1335 1343 1343 1348 1351 1359 1363 1373 1382 1417 1417 1430 1458 1458 1464 1464 1466 1472 1476 1477 1481 1506 1510 1545 1545 1571 1577 1579 1580 1580 1593 1595 1595 1605 1609 1609 1621 1708'.split()))
    # ans = convolutionCyclopeptideSequencing(spectrum, m, n)


    ##### problem 21 #####
    # money = 18142
    # coins = [1,3,5,15,17,19,24]
    # ans = changeProblem(money, coins)


    ##### problem 22 #####
    # data = open('../../Downloads/rosalind_ba5b (3).txt').read().splitlines()
    # n, m = list(map(int, data[0].split()))
    # down = [list(map(int, data[i].split())) for i in range(1, n + 1)]
    # right = [list(map(int, data[i].split())) for i in range(n + 2, 2 * n + 3)]
    # ans = manhattanTourist(n, m, down, right)


    ##### problem 23 #####
    # data = open('../../Downloads/rosalind_ba5e (11).txt').read().splitlines()
    # aaStr1 = data[0]
    # aaStr2 = data[1]
    # ans = globalAlignment(aaStr1, aaStr2)


    ##### problem 24 #####
    # data = open('../../Downloads/rosalind_ba5f (16).txt').read().splitlines()
    # aaStr1 = data[0]
    # aaStr2 = data[1]
    # ans = localAlignment(aaStr1, aaStr2)


    ##### problem 25 #####
    # data = open('../../Downloads/rosalind_ba5m (2).txt').read().splitlines()
    # ans = multipleLongestCommonSubsequence(*data)


    ##### problem 26 #####
    # permutation = '(-39 -123 +101 +110 -36 -94 +64 +134 -87 +73 +76 -21 -137 -65 -88 +23 +60 +80 -7 +78 -63 -22 -104 +107 +126 -93 +125 +2 +122 +131 +49 -118 +37 +25 -3 -8 -121 -47 +82 +18 +61 +103 +14 -57 +85 +46 -50 +26 +79 -109 +129 +27 -17 +41 -115 +108 -99 -140 -33 -72 +74 +70 -75 +141 -66 +43 +130 -42 +1 -53 +16 +71 +124 -24 +86 +29 +127 +77 -35 -9 -20 +112 -51 -133 -135 +81 -67 -132 -4 -106 +52 -116 +15 -98 -28 +13 -34 -30 -38 +48 +114 +117 +102 -10 +58 +95 -54 -45 +90 +138 -6 -44 +5 +83 -12 +19 +40 -136 +113 -111 -59 +89 -91 -105 -55 -139 -69 -119 +97 +120 +92 -56 +96 +62 -84 -31 +11 +128 +100 -68 -32)'
    # ans = greedySorting([int(x) for x in re.sub(r'[()]', '', permutation).split()])
    # print(*[f"({' '.join([f'+{a}' if a > 0 else str(a) for a in row])})" for row in ans], sep='\n')


    ##### problem 27 #####
    # permutation = open('../../Downloads/rosalind_ba6b (2).txt').read()
    # ans = numberOfBreaks([int(x) for x in re.sub(r'[()]', '', permutation).split()])


    ##### problem 28 #####
    # data = open('../../Downloads/rosalind_ba6c (3).txt').read().splitlines()
    # p = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in data[0].split(')(')]
    # q = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in data[1].split(')(')]
    # ans = twoBreakDistance(p, q)


    # chromosome = '(+1 +2 +3 -4 +5 -6 +7 -8 +9 +10 -11 -12 +13 -14 +15 +16 +17 +18 +19 -20 -21 -22 +23 +24 +25 +26 +27 +28 +29 +30 +31 -32 +33 -34 +35 +36 +37 -38 +39 +40 -41 -42 -43 -44 -45 -46 -47 +48 +49 +50 -51 -52 -53 -54 +55 -56 +57 -58 -59 +60 +61 +62 -63 +64)'
    # ans = chromosomeToCycle([int(x) for x in re.sub(r'[()]', '', chromosome).split()])


    # nodes = '(2 1 3 4 6 5 7 8 10 9 12 11 14 13 16 15 17 18 20 19 22 21 24 23 26 25 27 28 29 30 31 32 33 34 36 35 37 38 39 40 41 42 44 43 45 46 47 48 49 50 52 51 54 53 56 55 57 58 59 60 61 62 64 63 65 66 67 68 70 69 72 71 73 74 75 76 77 78 80 79 81 82 83 84 86 85 87 88 89 90 92 91 94 93 95 96 97 98 100 99 102 101 104 103 105 106 107 108 109 110 111 112 114 113 116 115 117 118 119 120)'
    # ans = cycleToChromosome([int(x) for x in re.sub(r'[()]', '', nodes).split()])


    # P = '(-1 +2 +3 +4 -5 +6 +7 -8 +9 +10 +11 -12 -13 +14 +15 +16 +17 -18 -19 -20 +21 -22)(+23 -24 +25 +26 -27 +28 -29 +30 +31 -32 -33 +34 +35 +36 +37 -38 -39 -40 +41 +42 -43 -44 -45 -46 +47 +48 -49)(-50 +51 +52 -53 -54 -55 +56 -57 +58 -59 -60 -61 -62 -63 -64 +65 +66 -67 -68 +69 -70 +71 -72)(-73 +74 -75 +76 +77 -78 +79 +80 -81 +82 -83 -84 +85 +86 +87 -88 -89 -90 -91 +92 +93 +94 -95 +96 -97 -98 -99 -100)(-101 +102 +103 +104 -105 -106 +107 -108 -109 +110 -111 -112 -113 -114 +115 +116 +117 -118 -119 +120 +121 +122 +123 +124 +125)(-126 -127 -128 +129 +130 -131 +132 -133 -134 +135 -136 +137 +138 -139 -140 +141 +142 -143 +144 -145 -146 +147 +148 -149 +150 +151 +152 -153 +154)(+155 +156 +157 -158 -159 +160 -161 +162 +163 -164 -165 -166 -167 +168 +169 -170 +171 -172 -173 -174 -175 -176 +177 -178)'
    # P = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in P.split(')(')]
    # ans = coloredEdges(P)


    # genomeGraph = '(2, 3), (4, 6), (5, 8), (7, 9), (10, 12), (11, 14), (13, 15), (16, 18), (17, 19), (20, 21), (22, 24), (23, 25), (26, 27), (28, 29), (30, 31), (32, 34), (33, 35), (36, 37), (38, 40), (39, 41), (42, 43), (44, 45), (46, 47), (48, 49), (50, 1), (52, 53), (54, 55), (56, 57), (58, 60), (59, 62), (61, 63), (64, 65), (66, 68), (67, 69), (70, 72), (71, 74), (73, 75), (76, 78), (77, 80), (79, 82), (81, 83), (84, 85), (86, 87), (88, 89), (90, 92), (91, 94), (93, 95), (96, 97), (98, 99), (100, 102), (101, 104), (103, 105), (106, 108), (107, 51), (110, 112), (111, 114), (113, 115), (116, 117), (118, 120), (119, 121), (122, 124), (123, 126), (125, 128), (127, 129), (130, 132), (131, 133), (134, 136), (135, 138), (137, 139), (140, 141), (142, 143), (144, 145), (146, 148), (147, 150), (149, 152), (151, 153), (154, 155), (156, 157), (158, 159), (160, 162), (161, 164), (163, 109), (166, 167), (168, 170), (169, 171), (172, 174), (173, 175), (176, 178), (177, 180), (179, 181), (182, 184), (183, 186), (185, 187), (188, 189), (190, 191), (192, 193), (194, 195), (196, 197), (198, 199), (200, 201), (202, 203), (204, 206), (205, 208), (207, 210), (209, 211), (212, 214), (213, 165), (215, 218), (217, 220), (219, 222), (221, 224), (223, 226), (225, 227), (228, 230), (229, 231), (232, 233), (234, 236), (235, 237), (238, 240), (239, 241), (242, 243), (244, 246), (245, 247), (248, 249), (250, 252), (251, 253), (254, 255), (256, 258), (257, 260), (259, 261), (262, 264), (263, 265), (266, 216), (267, 270), (269, 271), (272, 274), (273, 275), (276, 278), (277, 279), (280, 281), (282, 284), (283, 286), (285, 287), (288, 290), (289, 292), (291, 293), (294, 295), (296, 297), (298, 299), (300, 302), (301, 303), (304, 305), (306, 308), (307, 309), (310, 268), (311, 313), (314, 316), (315, 317), (318, 320), (319, 322), (321, 324), (323, 326), (325, 328), (327, 330), (329, 332), (331, 334), (333, 336), (335, 337), (338, 340), (339, 342), (341, 344), (343, 345), (346, 347), (348, 349), (350, 352), (351, 353), (354, 355), (356, 358), (357, 359), (360, 362), (361, 364), (363, 365), (366, 312)'
    # genomeGraph = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in genomeGraph.replace(',', '').split(') (')]
    # ans = graphToGenome(genomeGraph)


    # genomeGraph = '(2, 3), (4, 5), (6, 7), (8, 10), (9, 12), (11, 13), (14, 16), (15, 18), (17, 19), (20, 22), (21, 24), (23, 26), (25, 28), (27, 30), (29, 32), (31, 34), (33, 36), (35, 37), (38, 39), (40, 41), (42, 44), (43, 45), (46, 48), (47, 50), (49, 52), (51, 53), (54, 55), (56, 58), (57, 60), (59, 62), (61, 64), (63, 66), (65, 68), (67, 69), (70, 71), (72, 74), (73, 75), (76, 78), (77, 80), (79, 82), (81, 84), (83, 85), (86, 88), (87, 89), (90, 92), (91, 94), (93, 96), (95, 98), (97, 99), (100, 102), (101, 104), (103, 106), (105, 107), (108, 109), (110, 112), (111, 113), (114, 116), (115, 117), (118, 119), (120, 121), (122, 124), (123, 126), (125, 127), (128, 1)'
    # genomeGraph = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in genomeGraph.replace(',', '').split(') (')]
    # i, i_, j, j_ = 128, 1, 2, 3
    # ans = twoBreakOnGenomeGraph(genomeGraph, i, i_, j, j_)


    ##### problem 29 #####
    # data = open('../../Downloads/rosalind_ba6d (9).txt').read().splitlines()
    # # data = ['(+1 -2 -3 +4)', '(+1 +2 -4 -3)']
    # p = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in data[0].split(')(')]
    # q = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in data[1].split(')(')]
    # ans = twoBreakSorting(p, q)


    ##### problem 30 #####
    P = '(-1 -2 +3 +4 +5 +6 -7 -8 +9 +10 -11 -12 -13 -14 -15 -16 -17 -18 +19 -20 -21 -22 -23 +24 +25 -26 -27 -28 +29 -30 -31 +32 -33 -34 +35 -36 -37 +38 -39 -40 +41 -42 +43 -44 -45 +46 +47 -48 +49 +50 -51 -52 -53 -54 +55 +56 -57 -58 +59 -60 -61 -62 -63 +64 -65)'
    P = [[int(x) for x in re.sub(r'[()]', '', d).split()] for d in P.split(')(')]
    i, i_, j, j_ = 101, 104, 86, 88
    ans = twoBreakOnGenome(P, i, i_, j, j_)

    print(*[f"({' '.join([f'+{a}' if a > 0 else str(a) for a in r])})" for r in ans], sep=' ')
