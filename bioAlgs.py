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
        print('{0:b}'.format(i))
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
    data = open('../../Downloads/rosalind_ba5m (2).txt').read().splitlines()
    ans = multipleLongestCommonSubsequence(*data)

    print(*ans, sep='\n')
