import os
import sys

from deepeval.models import GeminiModel

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase
from agent_instrumented import support_agent

judge_model = GeminiModel(
model="gemini-3.6-flash"
)

result= support_agent("Where is my order ORD-1042?")

actual_output= result[0]["text"]

print(type(actual_output))
print(actual_output)

test_case = LLMTestCase(
    input = "Where is my order ORD-1042?",
    actual_output = actual_output
)

metric = TaskCompletionMetric(threshold=0.7,model = judge_model,include_reason=True)

evaluate(test_cases = [test_case],
         metrics= [metric])

# metric.measure(test_case)

#print("Score:", metric.score)
#print("Reason:", metric.reason)
#print("Success:", metric.success)