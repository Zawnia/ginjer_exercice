import os
from langfuse import Langfuse

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-7420c6ac-a6f1-42e6-be8d-96acba9f44b7"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-632ebd5e-3628-401d-8907-acbac25b2fbc"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

lf = Langfuse()
with lf.start_as_current_observation(name="test") as obs:
    print(dir(obs))
