using Microsoft.AspNetCore.Mvc;
using SteamRecommenderAPI.Services;

namespace SteamRecommenderAPI.Controllers;

[ApiController]
[Route("[controller]")]
public class RecommendController : ControllerBase
{
    private readonly SteamApiService _steamApiService;
    private readonly EmbeddingService _embeddingService;
    private readonly RecommendationService _recommendationService;

    public RecommendController(
        SteamApiService steamApiService, 
        EmbeddingService embeddingService, 
        RecommendationService recommendationService)
    {
        _steamApiService = steamApiService;
        _embeddingService = embeddingService;
        _recommendationService = recommendationService;
    }

    [HttpGet]
    public async Task<IActionResult> Get([FromQuery] string steam_id)
    {
        if (string.IsNullOrEmpty(steam_id))
            return BadRequest("steam_id is required.");

        try
        {
            // 1. Fetch user data from Steam API
            var userGames = await _steamApiService.GetUserGamesAsync(steam_id);
            var ownedAppIds = userGames.Select(g => g.AppId).ToList();

            // 2. Generate User Embedding from ML API (sending raw owned games)
            var userEmbedding = await _embeddingService.GetUserEmbeddingAsync(userGames);

            if (userEmbedding == null || !userEmbedding.Any())
                return StatusCode(500, "Failed to generate user embedding.");

            // 4. Calculate Recommendations
            var recommendations = await _recommendationService.GetRecommendationsAsync(userEmbedding, ownedAppIds ,20);

            return Ok(recommendations);
        }
        catch (Exception ex)
        {
            return StatusCode(500, ex.Message);
        }
    }
}
