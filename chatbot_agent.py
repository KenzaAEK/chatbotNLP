"""
CHATBOT IA AVEC NLP - VERSION OLLAMA 
==============================================
Agent conversationnel intelligent avec modèle local (pas besoin d'API payante)

Installation requise:
pip install langchain langchain-community
pip install spacy nltk chromadb
pip install streamlit python-dotenv ollama
python -m spacy download fr_core_news_md

Installation Ollama:
curl -fsSL https://ollama.com/install.sh | sh  (Linux/Mac)
ou téléchargez depuis https://ollama.com (Windows)

Puis: ollama pull mistral
==============================================
Realisé par ABOU-EL KASEM KENZA - Novembre 2025
"""

import os
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Télécharger les ressources NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# ============================================================================
# 1. CLASSE NLP - Traitement du langage naturel
# ============================================================================

class NLPProcessor:
    """Traite le texte avec analyse d'entités, sentiment et intentions"""
    
    def __init__(self, language='fr'):
        try:
            self.nlp = spacy.load('fr_core_news_md')
            print(" Modèle français chargé")
        except:
            print(" Modèle français non trouvé. Installation de l'anglais...")
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                print(" Aucun modèle spaCy trouvé. Téléchargez avec:")
                print("python -m spacy download fr_core_news_md")
                self.nlp = None
        
        self.sia = SentimentIntensityAnalyzer()
        
    def extract_entities(self, text):
        """Extrait les entités nommées (personnes, lieux, organisations)"""
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    
    def analyze_sentiment(self, text):
        """Analyse le sentiment du texte"""
        scores = self.sia.polarity_scores(text)
        
        if scores['compound'] >= 0.05:
            sentiment = 'positif'
        elif scores['compound'] <= -0.05:
            sentiment = 'négatif'
        else:
            sentiment = 'neutre'
            
        return {
            'sentiment': sentiment,
            'score': scores['compound'],
            'details': scores
        }
    
    def classify_intent(self, text):
        """Classifie l'intention de l'utilisateur"""
        text_lower = text.lower()
        
        # Règles simples de classification d'intention
        if any(word in text_lower for word in ['bonjour', 'salut', 'hello', 'hi', 'coucou']):
            return 'salutation'
        elif any(word in text_lower for word in ['au revoir', 'bye', 'à bientôt', 'adieu']):
            return 'au_revoir'
        elif any(word in text_lower for word in ['?', 'comment', 'pourquoi', 'quoi', 'quel', 'qui', 'où', 'quand']):
            return 'question'
        elif any(word in text_lower for word in ['aide', 'help', 'assistance', 'aidez-moi']):
            return 'aide'
        elif any(word in text_lower for word in ['merci', 'thank', 'remercie']):
            return 'remerciement'
        else:
            return 'conversation'
    
    def preprocess(self, text):
        """Nettoie et prépare le texte"""
        if not self.nlp:
            return {
                'clean_text': text.lower(),
                'original': text,
                'tokens': text.split()
            }
            
        doc = self.nlp(text)
        
        # Lemmatisation et nettoyage
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct]
        
        return {
            'clean_text': ' '.join(tokens),
            'original': text,
            'tokens': tokens
        }

# ============================================================================
# 2. CLASSE CHATBOT - Agent conversationnel avec Ollama
# ============================================================================

class ChatbotAgent:
    """Agent conversationnel intelligent avec Ollama (modèle local gratuit)"""
    
    def __init__(self, model_name="mistral", temperature=0.7, base_url="http://localhost:11434"):
        """
        Initialise le chatbot avec Ollama
        
        Modèles recommandés:
        - mistral: Équilibré, bon en français (7B)
        - llama2: Performant, anglais principalement (7B)
        - phi: Léger et rapide (2.7B)
        - neural-chat: Optimisé pour conversation (7B)
        - openchat: Bon pour le dialogue (7B)
        """
        
        print(f" Initialisation du modèle {model_name}...")
        
        try:
            # Initialiser Ollama
            self.llm = Ollama(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                # callbacks=[StreamingStdOutCallbackHandler()]  # Pour streaming en temps réel
            )
            
            # Test rapide du modèle
            test_response = self.llm.invoke("Bonjour")
            print(f" Modèle {model_name} chargé et fonctionnel")
            
        except Exception as e:
            print(f" Erreur lors du chargement du modèle: {e}")
            print("\n Vérifications:")
            print("1. Ollama est-il installé ? Tapez: ollama --version")
            print("2. Le service est-il lancé ? Tapez: ollama serve")
            print(f"3. Le modèle est-il téléchargé ? Tapez: ollama pull {model_name}")
            print("\n Modèles disponibles: ollama list")
            raise
        
        # Mémoire conversationnelle
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Processeur NLP
        self.nlp_processor = NLPProcessor()
        
        # Template de prompt optimisé pour modèles locaux
        template = """Tu es un assistant IA serviable, amical et concis. Tu réponds en français de manière naturelle.

Historique de conversation:
{chat_history}

Utilisateur: {input}
Assistant:"""
        
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template=template
        )
        
        # Chaîne de conversation
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )
        
        # Statistiques
        self.stats = {
            'total_messages': 0,
            'sentiments': {'positif': 0, 'négatif': 0, 'neutre': 0},
            'intents': {}
        }
        
        self.model_name = model_name
    
    def analyze_input(self, user_input):
        """Analyse complète du message utilisateur"""
        analysis = {
            'entities': self.nlp_processor.extract_entities(user_input),
            'sentiment': self.nlp_processor.analyze_sentiment(user_input),
            'intent': self.nlp_processor.classify_intent(user_input),
            'preprocessed': self.nlp_processor.preprocess(user_input)
        }
        
        # Mise à jour des statistiques
        self.stats['total_messages'] += 1
        sentiment = analysis['sentiment']['sentiment']
        self.stats['sentiments'][sentiment] += 1
        
        intent = analysis['intent']
        self.stats['intents'][intent] = self.stats['intents'].get(intent, 0) + 1
        
        return analysis
    
    def generate_response(self, user_input, show_analysis=False):
        """Génère une réponse avec analyse NLP optionnelle"""
        # Analyser l'entrée
        analysis = self.analyze_input(user_input)
        
        # Adapter la réponse selon l'intention
        if analysis['intent'] == 'salutation':
            context_hint = "L'utilisateur te salue. Réponds chaleureusement en une phrase."
        elif analysis['intent'] == 'au_revoir':
            context_hint = "L'utilisateur dit au revoir. Termine poliment en une phrase."
        elif analysis['intent'] == 'aide':
            context_hint = "L'utilisateur demande de l'aide. Sois clair et utile."
        else:
            context_hint = ""
        
        # Enrichir le prompt avec le contexte NLP
        enriched_input = user_input
        if context_hint:
            enriched_input = f"{context_hint}\n{user_input}"
        
        try:
            # Générer la réponse
            response = self.conversation.predict(input=enriched_input)
            
            # Nettoyer la réponse (enlever les répétitions parfois générées par les modèles locaux)
            response = response.strip()
            
        except Exception as e:
            response = f"Désolé, j'ai rencontré une erreur: {str(e)}"
        
        result = {'response': response}
        
        if show_analysis:
            result['analysis'] = analysis
        
        return result
    
    def get_stats(self):
        """Retourne les statistiques de conversation"""
        return self.stats
    
    def clear_memory(self):
        """Réinitialise la mémoire de conversation"""
        self.memory.clear()
        self.stats = {
            'total_messages': 0,
            'sentiments': {'positif': 0, 'négatif': 0, 'neutre': 0},
            'intents': {}
        }
    
    def get_model_info(self):
        """Retourne les informations sur le modèle"""
        return {
            'model': self.model_name,
            'type': 'Ollama (Local)',
            'cost': 'Gratuit',
            'privacy': '100% Local'
        }

# ============================================================================
# 3. INTERFACE LIGNE DE COMMANDE
# ============================================================================

def run_cli_chatbot():
    """Lance le chatbot en mode console"""
    print("=" * 60)
    print(" CHATBOT IA AVEC NLP (Version Ollama - Gratuite)")
    print("=" * 60)
    print("\nCommandes spéciales:")
    print("  /stats    - Afficher les statistiques")
    print("  /analyze  - Activer/désactiver l'analyse NLP")
    print("  /clear    - Effacer la mémoire")
    print("  /model    - Changer de modèle")
    print("  /info     - Informations sur le modèle")
    print("  /quit     - Quitter\n")
    
    # Demander quel modèle utiliser
    print(" Modèles Ollama disponibles:")
    print("  1. mistral (recommandé, bon en français)")
    print("  2. llama2 (performant)")
    print("  3. phi (léger et rapide)")
    print("  4. neural-chat (optimisé conversation)")
    
    model_choice = input("\nChoisir le modèle [1-4] ou appuyez sur Entrée pour mistral: ").strip()
    
    models = {
        '1': 'mistral',
        '2': 'llama2',
        '3': 'phi',
        '4': 'neural-chat',
        '': 'mistral'
    }
    
    model_name = models.get(model_choice, 'mistral')
    
    # Créer l'agent
    try:
        agent = ChatbotAgent(model_name=model_name)
    except Exception as e:
        print(f"\n Impossible de démarrer le chatbot.")
        print("\n Installation rapide:")
        print("1. Installez Ollama: https://ollama.com")
        print(f"2. Téléchargez le modèle: ollama pull {model_name}")
        print("3. Relancez ce script")
        return
    
    show_analysis = False
    
    print("\n Chatbot prêt ! Commencez à parler...\n")
    
    while True:
        try:
            user_input = input("Vous: ").strip()
            
            if not user_input:
                continue
            
            # Commandes spéciales
            if user_input.startswith('/'):
                if user_input == '/quit':
                    print("\n Au revoir !")
                    break
                    
                elif user_input == '/stats':
                    stats = agent.get_stats()
                    print(f"\n Statistiques:")
                    print(f"  Messages: {stats['total_messages']}")
                    print(f"  Sentiments: {stats['sentiments']}")
                    print(f"  Intentions: {stats['intents']}\n")
                    continue
                    
                elif user_input == '/analyze':
                    show_analysis = not show_analysis
                    status = "activée" if show_analysis else "désactivée"
                    print(f"\n Analyse NLP {status}\n")
                    continue
                    
                elif user_input == '/clear':
                    agent.clear_memory()
                    print("\n Mémoire effacée\n")
                    continue
                    
                elif user_input == '/info':
                    info = agent.get_model_info()
                    print(f"\n Informations:")
                    print(f"  Modèle: {info['model']}")
                    print(f"  Type: {info['type']}")
                    print(f"  Coût: {info['cost']}")
                    print(f"  Confidentialité: {info['privacy']}\n")
                    continue
                    
                elif user_input == '/model':
                    print("\n Pour changer de modèle, relancez le programme")
                    print("ou utilisez: ollama run <nom_modèle>\n")
                    continue
            
            # Générer la réponse
            print("\nRéflexion...", end="\r")
            result = agent.generate_response(user_input, show_analysis)
            print(" " * 20, end="\r")  # Effacer le message
            
            # Afficher l'analyse si activée
            if show_analysis and 'analysis' in result:
                analysis = result['analysis']
                print(f"\n Analyse:")
                print(f"  Sentiment: {analysis['sentiment']['sentiment']} ({analysis['sentiment']['score']:.2f})")
                print(f"  Intention: {analysis['intent']}")
                if analysis['entities']:
                    print(f"  Entités: {[e['text'] + ' (' + e['label'] + ')' for e in analysis['entities']]}")
            
            # Afficher la réponse
            print(f"\nBot: {result['response']}\n")
            
        except KeyboardInterrupt:
            print("\n\n Au revoir !")
            break
        except Exception as e:
            print(f"\n Erreur: {e}\n")

# ============================================================================
# 4. POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    print("\n Bienvenue ! Ce chatbot utilise Ollama (100% gratuit et local)")
    print("Aucune clé API nécessaire - Vos données restent privées\n")
    
    run_cli_chatbot()