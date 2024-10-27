1. The readme instructions were wrong, see fixed paths.
2. `requirements.txt` was wrong, it included a lot of packages specific for your machine. Did you generate it using `pip freeze' or `pip list`? See the new requirements which only has top-level requirements, installing these installs the required packages specific for each machine / OS.
3. Never use “from X import *”, instead specify exactly what you’re importing, much easier to keep track of where function calls are made, and you’re also defending yourself against running code that you never intended to run in that package (e.g., variable assignment  or loose code)
4. Saving figures: better to take as command line arguments rather than hardcoded ones.

