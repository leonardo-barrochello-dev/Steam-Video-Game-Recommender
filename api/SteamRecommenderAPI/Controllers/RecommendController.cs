using Microsoft.AspNetCore.Mvc;
using SteamRecommenderAPI.Services;

namespace SteamRecommenderAPI.Controllers;

[ApiController]
[Route("[controller]")]
public class RecommendController : ControllerBase
{
    private readonly SteamApiService _steamApiService;
    private readonly RecommendService _recommendService;

    public RecommendController(
        SteamApiService steamApiService,
        RecommendService recommendService)
    {
        _steamApiService = steamApiService;
        _recommendService = recommendService;
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

            if (userGames == null || !userGames.Any())
                return StatusCode(500, "No games found for this Steam ID.");

            // 2. Get recommendations from Python ML API (embedding + Qdrant + filter)
            var recommendations = await _recommendService.GetRecommendationsAsync(userGames, topK: 10);

            return Ok(recommendations);
        }
        catch (Exception ex)
        {
            return StatusCode(500, ex.Message);
        }
    }
}
