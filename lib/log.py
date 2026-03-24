from datetime import datetime

_template = "%t %m"

def configure(template: str):
    global _template
    _template = template

def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = _template.replace("%t", now).replace("%m", msg)
    print(output, **kwargs)
