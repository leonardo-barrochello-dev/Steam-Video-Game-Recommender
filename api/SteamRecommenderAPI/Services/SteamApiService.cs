using System.Text.Json;
using System.Text.Json.Serialization;

namespace SteamRecommenderAPI.Services;

public class SteamGame
{
    [JsonPropertyName("appid")]
    public int AppId { get; set; }

    [JsonPropertyName("playtime_forever")]
    public int PlaytimeForever { get; set; }
}

public class SteamResponse
{
    [JsonPropertyName("games")]
    public List<SteamGame> Games { get; set; } = new();
}

public class SteamRoot
{
    [JsonPropertyName("response")]
    public SteamResponse Response { get; set; } = new();
}

public class SteamApiService
{
    private readonly HttpClient _httpClient;
    private readonly IConfiguration _configuration;

    public SteamApiService(HttpClient httpClient, IConfiguration configuration)
    {
        _httpClient = httpClient;
        _configuration = configuration;
    }

    public async Task<List<SteamGame>> GetUserGamesAsync(string steamId)
    {
        var apiKey = _configuration["Steam:ApiKey"];
        if (string.IsNullOrEmpty(apiKey) || apiKey == "REPLACE_WITH_YOUR_STEAM_API_KEY")
        {
            throw new Exception("Steam API Key is not configured properly.");
        }

        var url = $"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={apiKey}&steamid={steamId}&format=json";
        
        var response = await _httpClient.GetAsync(url);
        response.EnsureSuccessStatusCode();

        var jsonStr = await response.Content.ReadAsStringAsync();
        var root = JsonSerializer.Deserialize<SteamRoot>(jsonStr);

        return root?.Response?.Games ?? new List<SteamGame>();
    }
}
