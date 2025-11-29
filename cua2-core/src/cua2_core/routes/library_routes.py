from fastapi import APIRouter
from ..services.library_service import LibraryService

router = APIRouter(prefix="/api/library", tags=["library"])

@router.get("/nodes")
async def get_standard_nodes():
    """Get available standard node templates."""
    return LibraryService.get_standard_nodes()

@router.get("/agents")
async def get_agent_types():
    """Get available agent types."""
    return LibraryService.get_agent_types()
