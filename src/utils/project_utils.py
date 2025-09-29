from pathlib import Path

class ProjectUtils:
    
    @staticmethod
    def get_project_root() -> Path:
        """
        Returns the project root directory.
        """
        return Path(__file__).resolve().parent.parent.parent