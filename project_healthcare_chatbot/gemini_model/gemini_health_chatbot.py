from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

import warnings
warnings.filterwarnings('ignore')

class MedBot:
  def __init__(self, google_api_key="secret_key", model="gemini-2.0-flash", temperature=0.3, top_k=5, top_p=0.5):
    self.model = ChatGoogleGenerativeAI(
      google_api_key=google_api_key,
      model=model,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p
    )

    self.system_prompt = """
            You are MedBot, an AI-powered healthcare assistant. Your role is to provide accurate and reliable medical information.
            Guidelines:
            1. Do not provide information outside the medical field. If asked, reply that you are a medical chatbot and also remember all previous chats with all domain.
            2. Answer only healthcare-related questions.
            3. Use verified medical sources.
            4. Explain complex medical terms in an easy-to-understand way.
            5. Do not provide personal diagnoses or prescribe medications.
            6. If the question requires emergency help, advise the user to consult a medical professional immediately.
            7. Maintain a polite and professional tone at all times.
            8. Avoid unnecessary information or long explanations.
            9. If the user asks for more details, provide them upon request.
        """

    self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    self.prompt_template = ChatPromptTemplate.from_messages([
      SystemMessage(content=self.system_prompt),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{user_input}")
    ])

    self.chat_chain = LLMChain(llm=self.model, prompt=self.prompt_template, memory=self.memory)

  def generate_response(self, user_input):
    response = self.chat_chain.run(user_input)
    return response