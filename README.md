# 🤖 Chatbot NLP & LLM Local (Ollama)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Ollama](https://img.shields.io/badge/AI-Ollama%20(Local)-orange)

Un assistant conversationnel intelligent, 100% local et respectueux de la vie privée.  
Ce projet combine la puissance des **LLMs génératifs** (via Ollama) avec une couche d'analyse **NLP classique** (spaCy, NLTK) pour offrir des réponses contextuelles et précises.

---

## 🚀 Fonctionnalités Clés

- **🔒 100% Local & Privé** : Aucune donnée ne quitte votre machine. Utilise Ollama pour faire tourner des modèles comme Mistral ou Llama2 en local.
- **🧠 Analyse NLP Hybride** :
  - **Détection d'entités (NER)** : Identifie les personnes, lieux et organisations (via spaCy).
  - **Analyse de Sentiment** : Évalue la tonalité des messages (via NLTK).
  - **Classification d'Intention** : Moteur heuristique pour les interactions rapides.
- **🎨 Interface Moderne** : Application Web interactive construite avec Streamlit.
- **📊 Tableau de Bord** : Visualisation en temps réel des statistiques de conversation (sentiments, métriques).

---

## 🛠️ Architecture Technique

Le projet suit une architecture modulaire :

1.  **Frontend** : Streamlit (`app.py`) pour l'interaction utilisateur.
2.  **Backend Logic** : Agent conversationnel (`chatbot_agent.py`) gérant la mémoire et le NLP.
3.  **Intelligence** : 
    - **Génératif** : Ollama (Mistral 7B).
    - **Analytique** : spaCy (`fr_core_news_md`) + NLTK Vader.

---

## 📦 Installation

### Prérequis
- Python 3.10 ou supérieur
- [Ollama](https://ollama.com/) installé et en cours d'exécution

### 1. Cloner le dépôt
```bash
git clone [https://github.com/KenzaAEK/chatbotNLP.git](https://github.com/KenzaAEK/chatbotNLP.git)
cd votre-repo
