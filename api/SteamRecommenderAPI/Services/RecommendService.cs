using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SteamRecommenderAPI.Services;

public class RecommendRequestItem
{
    [JsonPropertyName("appid")]
    public int AppId { get; set; }

    [JsonPropertyName("playtime_forever")]
    public int PlaytimeForever { get; set; }
}

public class RecommendRequest
{
    [JsonPropertyName("owned_games")]
    public List<RecommendRequestItem> OwnedGames { get; set; } = new();

    [JsonPropertyName("top_k")]
    public int TopK { get; set; } = 10;
}

public class RecommendCandidate
{
    [JsonPropertyName("app_id")]
    public int AppId { get; set; }

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("score")]
    public float Score { get; set; }
}

public class RecommendResponse
{
    [JsonPropertyName("recommendations")]
    public List<RecommendCandidate> Recommendations { get; set; } = new();
}

public class RecommendService
{
    private readonly HttpClient _httpClient;
    private readonly IConfiguration _configuration;

    public RecommendService(HttpClient httpClient, IConfiguration configuration)
    {
        _httpClient = httpClient;
        _configuration = configuration;
    }

    public async Task<List<RecommendCandidate>> GetRecommendationsAsync(List<SteamGame> ownedGames, int topK = 10)
    {
        var baseUrl = _configuration["PythonML:BaseUrl"];
        if (string.IsNullOrEmpty(baseUrl))
        {
            throw new Exception("Python ML Base URL is not configured.");
        }

        var url = $"{baseUrl}/recommend";

        var requestPayload = new RecommendRequest
        {
            OwnedGames = ownedGames.Select(g => new RecommendRequestItem
            {
                AppId = g.AppId,
                PlaytimeForever = g.PlaytimeForever,
            }).ToList(),
            TopK = topK,
        };

        var jsonContent = JsonSerializer.Serialize(requestPayload);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync(url, content);
        if (!response.IsSuccessStatusCode)
        {
            var errorBody = await response.Content.ReadAsStringAsync();
            throw new Exception($"Python ML API error ({(int)response.StatusCode}): {errorBody}");
        }

        var jsonResponse = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<RecommendResponse>(jsonResponse);

        return result?.Recommendations ?? new List<RecommendCandidate>();
    }
}
