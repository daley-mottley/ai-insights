import logging
from typing import List
from openai import OpenAI
import google.generativeai as genai
from cachetools import cached, TTLCache
from config import Config
from services.report_generator import ReportGenerator

class LLMService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Default to Gemini unless overridden in Config
        self.model_type = Config.LLM_MODEL.lower() if Config.LLM_MODEL else 'gemini'

        if self.model_type == 'openai' and Config.USE_OPENAI_API:
            self.openai_api_key = Config.OPENAI_API_KEY
            self.model = Config.OPENAI_MODEL or 'gpt-4o-mini'  # Default OpenAI model
            self.client = OpenAI(api_key=self.openai_api_key)
            self.logger.info(f"Initialized LLMService with OpenAI model: {self.model}")
        elif self.model_type == 'gemini':
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = Config.GEMINI_MODEL or 'gemini-1.5-pro'  # Default Gemini model
            self.client = genai.GenerativeModel(self.model)
            self.logger.info(f"Initialized LLMService with Gemini model: {self.model}")
        else:
            self.logger.info('LLM service is disabled or invalid model type specified.')
            self.client = None
            self.model = None

        self.cache = TTLCache(maxsize=100, ttl=300)  # Cache results for 5 minutes
        if self.client and self.model:
            self.report_generator = ReportGenerator(self)
        else:
            self.report_generator = None

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def generate_text(self, prompt: str) -> str:
        """Generate text using the selected model."""
        if not self.client:
            self.logger.error("Cannot generate text: LLM client is not initialized.")
            return "LLM service is disabled or not properly configured."

        try:
            if self.model_type == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
            elif self.model_type == 'gemini':
                response = self.client.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            self.logger.error(f"Error generating text with {self.model_type}: {e}", exc_info=True)
            raise

    def generate_report_content(self, industry: str, answers: List[str], user_name: str) -> str:
        """Delegate report generation to the ReportGenerator."""
        if not self.report_generator:
            self.logger.error("ReportGenerator not initialized; cannot generate report.")
            return "Report generation is unavailable due to LLM service configuration."
        return self.report_generator.generate_report_content(industry, answers, user_name)
