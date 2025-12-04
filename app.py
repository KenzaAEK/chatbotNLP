"""
INTERFACE WEB STREAMLIT POUR LE CHATBOT OLLAMA
===============================================
Lance avec: streamlit run app.py
"""

import streamlit as st
from chatbot_agent import ChatbotAgent
import time

# Configuration de la page
st.set_page_config(
    page_title="Chatbot IA Local (Ollama)",
    page_icon="🤖",
    layout="wide"
)

# Initialisation de l'état de session
if 'agent' not in st.session_state:
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.show_analysis = False
    st.session_state.model_loaded = False

# Titre et description
st.title("🤖 Chatbot IA avec NLP (Version Ollama - Gratuite)")
st.markdown("Assistant intelligent 100% local - Vos données restent privées ✅")

# Sidebar avec options
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélection du modèle
    st.subheader("📦 Modèle Ollama")
    
    models_info = {
        "mistral": "Équilibré, bon en français (7B) ⭐",
        "llama2": "Performant, anglais (7B)",
        "phi": "Léger et rapide (2.7B)",
        "neural-chat": "Optimisé conversation (7B)",
        "openchat": "Bon pour dialogue (7B)"
    }
    
    selected_model = st.selectbox(
        "Choisir le modèle",
        options=list(models_info.keys()),
        format_func=lambda x: f"{x} - {models_info[x]}"
    )
    
    # Température
    temperature = st.slider(
        "Température (créativité)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Plus haute = plus créatif, plus basse = plus précis"
    )
    
    # Bouton de chargement du modèle
    if st.button("🚀 Charger le modèle", type="primary"):
        with st.spinner(f"Chargement de {selected_model}..."):
            try:
                st.session_state.agent = ChatbotAgent(
                    model_name=selected_model,
                    temperature=temperature
                )
                st.session_state.model_loaded = True
                st.session_state.messages = []
                st.success(f"✅ Modèle {selected_model} chargé !")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                st.info("""
                **Installation Ollama:**
                1. Téléchargez depuis https://ollama.com
                2. Installez Ollama
                3. Ouvrez un terminal et tapez:
                   ```
                   ollama pull mistral
                   ```
                4. Relancez cette application
                """)
                st.session_state.model_loaded = False
    
    st.divider()
    
    # Options d'affichage
    st.header("🔍 Options")
    st.session_state.show_analysis = st.checkbox(
        "Afficher l'analyse NLP",
        value=st.session_state.show_analysis
    )
    
    # Bouton pour effacer l'historique
    if st.button("🗑️ Effacer la conversation"):
        if st.session_state.agent:
            st.session_state.agent.clear_memory()
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Informations sur le modèle
    if st.session_state.model_loaded and st.session_state.agent:
        st.header("📋 Informations")
        info = st.session_state.agent.get_model_info()
        st.write(f"**Modèle:** {info['model']}")
        st.write(f"**Type:** {info['type']}")
        st.write(f"**Coût:** {info['cost']}")
        st.write(f"**Confidentialité:** {info['privacy']}")
        
        st.divider()
        
        # Statistiques
        st.header("📊 Statistiques")
        stats = st.session_state.agent.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats['total_messages'])
        with col2:
            if stats['total_messages'] > 0:
                positive_pct = (stats['sentiments']['positif'] / stats['total_messages']) * 100
                st.metric("Positif", f"{positive_pct:.0f}%")
        
        # Graphique des sentiments
        if stats['total_messages'] > 0:
            st.subheader("Sentiments")
            sentiment_data = stats['sentiments']
            st.bar_chart(sentiment_data)
            
            if stats['intents']:
                st.subheader("Intentions")
                intent_data = stats['intents']
                st.bar_chart(intent_data)

# Zone principale
if not st.session_state.model_loaded:
    st.info("👈 Veuillez charger un modèle Ollama dans la barre latérale pour commencer")
    
    # Guide d'installation
    with st.expander("📚 Guide d'installation Ollama"):
        st.markdown("""
        ### Installation Ollama (Gratuit)
        
        **1. Télécharger Ollama:**
        - Windows/Mac: https://ollama.com/download
        - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
        
        **2. Télécharger un modèle:**
        ```bash
        ollama pull mistral
        ```
        
        **3. Vérifier l'installation:**
        ```bash
        ollama list
        ```
        
        **4. Lancer cette application**
        
        ### Modèles recommandés:
        - **mistral** (7B) - Meilleur compromis, bon en français
        - **llama2** (7B) - Performant mais principalement anglais
        - **phi** (2.7B) - Léger et rapide pour machines modestes
        - **neural-chat** (7B) - Optimisé pour la conversation
        
        ### Commandes utiles:
        - `ollama list` - Voir les modèles installés
        - `ollama pull <modèle>` - Télécharger un modèle
        - `ollama rm <modèle>` - Supprimer un modèle
        """)
    
    st.warning("⚠️ Assurez-vous qu'Ollama est installé et qu'un modèle est téléchargé")

else:
    # Zone de chat
    chat_container = st.container()
    
    # Afficher l'historique des messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Afficher l'analyse si disponible
                if st.session_state.show_analysis and "analysis" in message:
                    with st.expander("🔍 Analyse NLP"):
                        analysis = message["analysis"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment = analysis['sentiment']['sentiment']
                            score = analysis['sentiment']['score']
                            
                            # Emoji selon le sentiment
                            emoji = "😊" if sentiment == "positif" else "😐" if sentiment == "neutre" else "😔"
                            st.write(f"**Sentiment:** {emoji} {sentiment}")
                            st.write(f"**Score:** {score:.2f}")
                        with col2:
                            intent = analysis['intent']
                            intent_emoji = {
                                'salutation': '👋',
                                'au_revoir': '👋',
                                'question': '❓',
                                'aide': '🆘',
                                'remerciement': '🙏',
                                'conversation': '💬'
                            }
                            st.write(f"**Intention:** {intent_emoji.get(intent, '💬')} {intent}")
                        
                        if analysis['entities']:
                            st.write("**Entités détectées:**")
                            for entity in analysis['entities']:
                                st.write(f"- {entity['text']} ({entity['label']})")
    
    # Input utilisateur
    if prompt := st.chat_input("Écrivez votre message..."):
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Générer la réponse
        with st.chat_message("assistant"):
            with st.spinner("🤔 Réflexion en cours..."):
                result = st.session_state.agent.generate_response(
                    prompt,
                    show_analysis=st.session_state.show_analysis
                )
                
                st.write(result['response'])
                
                # Afficher l'analyse
                if st.session_state.show_analysis and 'analysis' in result:
                    with st.expander("🔍 Analyse NLP"):
                        analysis = result['analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment = analysis['sentiment']['sentiment']
                            score = analysis['sentiment']['score']
                            
                            emoji = "😊" if sentiment == "positif" else "😐" if sentiment == "neutre" else "😔"
                            st.write(f"**Sentiment:** {emoji} {sentiment}")
                            st.write(f"**Score:** {score:.2f}")
                        with col2:
                            intent = analysis['intent']
                            intent_emoji = {
                                'salutation': '👋',
                                'au_revoir': '👋',
                                'question': '❓',
                                'aide': '🆘',
                                'remerciement': '🙏',
                                'conversation': '💬'
                            }
                            st.write(f"**Intention:** {intent_emoji.get(intent, '💬')} {intent}")
                        
                        if analysis['entities']:
                            st.write("**Entités détectées:**")
                            for entity in analysis['entities']:
                                st.write(f"- {entity['text']} ({entity['label']})")
        
        # Ajouter à l'historique
        message_data = {"role": "assistant", "content": result['response']}
        if 'analysis' in result:
            message_data['analysis'] = result['analysis']
        st.session_state.messages.append(message_data)
        
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Chatbot 100% local et gratuit - Propulsé par Ollama + LangChain + spaCy</small><br>
    <small>Vos données ne quittent jamais votre machine 🔒</small>
</div>
""", unsafe_allow_html=True)