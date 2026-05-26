# 🎮 Prompt — Sistema de Recomendação de Jogos (Steam)

## Two-Tower (TensorFlow) + API C# + SQLite + Steam API (Tempo Real)

Crie um sistema completo de recomendação de jogos baseado na Steam utilizando arquitetura Two-Tower, com treinamento em Python, serving em C# e integração direta com a Steam API para obter dados reais dos usuários.

---

## 🎯 Objetivo

Desenvolver um sistema que:

* Recomenda jogos personalizados para usuários reais da Steam
* Usa dados reais via Steam API
* Utiliza embeddings (Two-Tower)
* Armazena embeddings em SQLite
* Retorna recomendações em tempo real via API C#

---

## 🧱 Arquitetura Geral

### 🔵 ML Service (Python)

Responsável por:

* Pré-processamento dos datasets
* Treinamento do modelo Two-Tower
* Geração de embeddings dos jogos
* Exportação para SQLite

---

### 🟢 API (C# .NET)

Responsável por:

* Receber steam_id
* Consumir Steam API
* Construir features do usuário em tempo real
* Gerar user_embedding (via Python ou lógica compartilhada)
* Buscar jogos similares
* Retornar recomendações

---

### 🗄️ Banco (SQLite)

Armazenar:

* game_embeddings
* metadados dos jogos

---

## 📊 Datasets (Treinamento)

### steam_games.csv

* app_id
* name
* genres
* tags
* price
* release_date

---

### steam_reviews.csv

* user_id
* app_id
* playtime

---

## ⚙️ Pipeline (Python)

### 1. Merge

* Unir por app_id

---

### 2. Label

* playtime > 120 → 1
* caso contrário → 0

---

### 3. Negative Sampling

* gerar pares (user, jogo não jogado)

---

### 4. Feature Engineering

#### 🎮 Jogos

* genres → one-hot
* tags → multi-hot
* price normalizado
* release_year

---

#### 🧍 Usuários

* total_playtime
* avg_playtime
* preferência por gênero

---

## 🧠 Modelo (TensorFlow)

### User Tower

* Dense(256, relu)
* Dense(128, relu)
* Dense(embedding_dim)

---

### Item Tower

* Dense(256, relu)
* Dense(128, relu)
* Dense(embedding_dim)

---

### Embeddings

* L2 normalization

---

### Predição

* dot product + sigmoid

---

### Loss

* BinaryCrossentropy

---

## 🏋️ Treinamento

* Input: [user_features, item_features]
* Output: label

---

## 💾 Exportação

Salvar embeddings dos jogos no SQLite:

```sql
CREATE TABLE game_embeddings (
  app_id INTEGER PRIMARY KEY,
  embedding BLOB,
  name TEXT,
  genres TEXT
);
```

---

## 🔌 Integração com Steam API (OBRIGATÓRIO)

Consumir:

* GetOwnedGames
* GetRecentlyPlayedGames

---

## 🧠 Construção do User Embedding (Tempo Real)

### Fluxo:

1. Receber steam_id
2. Buscar jogos do usuário via API
3. Mapear app_id → features dos jogos
4. Construir user_features:

   * soma de playtime
   * média de playtime
   * distribuição de gêneros

---

### Geração do embedding

Opções:

#### ✔️ Opção A (recomendada)

Chamar serviço Python:

POST /embed-user

#### ✔️ Opção B

Reimplementar User Tower em C#

---

## 🔮 Recomendação

1. Gerar user_embedding
2. Carregar embeddings dos jogos (SQLite)
3. Calcular similaridade (dot product)
4. Ordenar
5. Remover jogos já possuídos
6. Retornar Top-N

---

## 🌐 API (C#)

Endpoint:

GET /recommend?steam_id=XXX

---

## 📤 Resposta

[
{
"app_id": 123,
"name": "Game Name",
"score": 0.91
}
]

---

## ⚡ Otimizações

* Cache de user_embedding
* Cache de jogos do usuário
* Pré-carregar embeddings em memória
* Atualização periódica do modelo

---

## 📁 Estrutura

/project
/ml (Python)
/api (C# .NET)
/data
/database (SQLite)

---

## 🧪 Diferenciais

* Recomendação baseada em dados reais
* Cold start para novos jogos
* Explicabilidade básica
* Filtro por gênero/preço

---

## 📌 Regras

* Separar treino e inferência
* Evitar loops pesados
* Priorizar embeddings
* Sistema deve funcionar com usuário real

---

## 🎯 Entregável

* Modelo treinado
* Embeddings no SQLite
* API em C# integrada com Steam
* Recomendação funcionando com steam_id real

---

## 💡 Objetivo final

Construir um sistema real de recomendação de jogos com dados reais, pronto para evoluir para um produto SaaS.

---
