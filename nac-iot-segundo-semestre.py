import math

dados = []
# read mode
with open("/Users/felipe/Downloads/cancer.data", "r") as dataset:
    for instancia in dataset.readlines():
        x = instancia.replace('\n', '')
        dataPositions = x.split(',')
        onePosition = dataPositions[0] \
            .replace('no-recurrence-events', '0') \
            .replace('recurrence-events', '1')

        twoPosition = dataPositions[1] \
            .replace('10-19', '0.1') \
            .replace('20-29', '0.22') \
            .replace('30-39', '0.33') \
            .replace('40-49', '0.44') \
            .replace('50-59', '0.55') \
            .replace('60-69', '0.66') \
            .replace('70-79', '0.77') \
            .replace('80-89', '0.88') \
            .replace('90-99', '0.99')

        threePosition = dataPositions[2] \
            .replace('premeno', '0.33') \
            .replace('ge40', '0.66') \
            .replace('lt40', '0.99')

        fourPosition = dataPositions[3] \
            .replace('0-4', '0.083') \
            .replace('5-9', '0.166') \
            .replace('10-14', '0.249') \
            .replace('15-19', '0.332') \
            .replace('20-24', '0.415') \
            .replace('25-29', '0.498') \
            .replace('30-34', '0.581') \
            .replace('35-39', '0.664') \
            .replace('40-44', '0.747') \
            .replace('45-49', '0.830') \
            .replace('50-54', '0.913') \
            .replace('55-59', '0.996')

        fivePosition = dataPositions[4] \
            .replace('0-2', '0.076') \
            .replace('3-5', '0.152') \
            .replace('6-8', '0.228') \
            .replace('9-11', '0.304') \
            .replace('12-14', '0.380') \
            .replace('15-17', '0.456') \
            .replace('18-20', '0.532') \
            .replace('21-23', '0.608') \
            .replace('24-26', '0.684') \
            .replace('27-29', '0.760') \
            .replace('30-32', '0.836') \
            .replace('33-35', '0.912') \
            .replace('36-39', '0.988')

        sixPosition = dataPositions[5] \
            .replace('no', '0')\
            .replace('yes', '1')\
            .replace('?', '0')

        sevenPosition = dataPositions[6]\

        eightPosition = dataPositions[7] \
            .replace('right', '1.0') \
            .replace('left', '0.0')

        ninePosition = dataPositions[8] \
            .replace('left_up', '0.2') \
            .replace('left_low', '0.4') \
            .replace('right_up', '0.6') \
            .replace('right_low', '0.8') \
            .replace('central', '1') \
            .replace('?', '0')

        tenPosition = dataPositions[9] \
            .replace('no', '0') \
            .replace('yes', '1')


        dados.append([
            float(onePosition),
            float(twoPosition),
            float(threePosition),
            float(fourPosition),
            float(fivePosition),
            float(sixPosition),
            float(sevenPosition),
            float(eightPosition),
            float(ninePosition),
            float(tenPosition)
        ])

# imprimindo resultados
def info_dataset(dados, info=True):
    output1, output2 = 0, 0

    for item in dados:
        if item[0] == 0:
            output1 += 1
        else:
            output2 += 1
    if info:
        print("")
        print("___________________________")
        print("Total de dados", len(dados))
        print("Total de mulheres sem câncer ", output1)
        print("Total de mulheres com câncer ", output2)
        print("___________________________")

    return [len(dados), output1, output2]


info_dataset(dados)

# separando um conjunto para testes
porcetagem = 0.6

_, output1, output2 = info_dataset(dados, info=False)

treinamento = []
teste = []

max_output1 = int(porcetagem * output1)
max_output2 = int(porcetagem * output2)

total_output1 = 0
total_output2 = 0

for item in dados:
    if (total_output1 + total_output2) < (max_output1 + max_output2):
        treinamento.append(item)
        if item[0] == 1 and total_output1 < max_output1:
            total_output1 += 1
        else:
            total_output2 += 1
    else:
        teste.append(item)

print('Quantidade de Treinamento: ', len(treinamento))
print('Quantidade de Teste: ', len(teste))

#mostra a distancia entre dois pontos
def distancia_euclidiana(treinamento, nova_amostra):
    dimensao = len(treinamento)
    soma = 0
    for i in range(dimensao):
        soma += (treinamento[i] - nova_amostra[i]) ** 2
    return math.sqrt(soma)

#inicio KNN
def knn(treinamento, nova_amostra, k):
    distancias = {}
    tamanho_treino = len(treinamento)
    # essa funcao calcula a distancia euclidiana da nova amostra para todos os outros exemplos do produto
    for i in range(tamanho_treino):
        d = distancia_euclidiana(treinamento[i], nova_amostra)
        distancias[i] = d
    #indice dos vizinhos mais proximos
    k_vizinhos = sorted(distancias, key=distancias.get)[:k]

    qtd_output1 = 0
    qtd_output2 = 0
    # 
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1:
            qtd_output1 += 1
        else:
            qtd_output2 += 1
    
    # tomada de decisão sobre o amount de dados
    if qtd_output1 > qtd_output2:
        return 1
    else:
        return 0


acertos = 0
k = 5

# teste de eficiência do knn
for item in teste:
    classe = knn(treinamento, item, k)
    if item[-1] == classe:
        acertos += 1

print(" ")
print("Resultados do treinamento -", len(treinamento),"itens para treinamento.")
print("Total de testes", len(teste))
print("Total de acertos", acertos)
print("Porcentagem de acerto", 100 * acertos / len(teste))




