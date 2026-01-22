import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def extract_functions(code_str: str):
    """Parses Python code and returns a list of function bodies and signatures."""
    tree = parser.parse(bytes(code_str, "utf8"))
    functions = []
    
    # Query to find function definitions
    query = PY_LANGUAGE.query("""
        (function_definition
            name: (identifier) @fn_name
            parameters: (parameters) @params
            body: (block) @body)
    """)
    
    captures = query.captures(tree.root_node)
    # Logic to group captures into a list of snippets...
    # (Simplified for the structure)
    return captures

if __name__ == "__main__":
    sample_code = "def hello(name):\n    print(f'Hello {name}')"
    print(extract_functions(sample_code))