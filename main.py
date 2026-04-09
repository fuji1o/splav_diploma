from fastmcp import FastMCP

mcp = FastMCP("HelloServer")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello, {name}!"

@mcp.tool()
def hello_by_surname(surname: str) -> str:
    """
    Приветствие по фамилии.
    
    Args:
        surname: Фамилия человека
        
    Returns:
        Формальное приветствие с использованием фамилии
    """
    return f"Здравствуйте, уважаемый(ая) {surname}!"

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=3000)