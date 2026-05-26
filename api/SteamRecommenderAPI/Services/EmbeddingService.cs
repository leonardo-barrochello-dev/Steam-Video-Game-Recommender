using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SteamRecommenderAPI.Services;

public class EmbeddingRequest
{
    [JsonPropertyName("owned_games")]
    public List<SteamGame> OwnedGames { get; set; } = new();
}

public class EmbeddingResponse
{
    [JsonPropertyName("user_embedding")]
    public List<float> UserEmbedding { get; set; } = new();
}

public class EmbeddingService
{
    private readonly HttpClient _httpClient;
    private readonly IConfiguration _configuration;

    public EmbeddingService(HttpClient httpClient, IConfiguration configuration)
    {
        _httpClient = httpClient;
        _configuration = configuration;
    }

    public async Task<List<float>> GetUserEmbeddingAsync(List<SteamGame> ownedGames)
    {
        var baseUrl = _configuration["PythonML:BaseUrl"];
        if (string.IsNullOrEmpty(baseUrl))
        {
            throw new Exception("Python ML Base URL is not configured.");
        }

        var url = $"{baseUrl}/embed-user";
        
        var requestPayload = new EmbeddingRequest { OwnedGames = ownedGames };
        var jsonContent = JsonSerializer.Serialize(requestPayload);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync(url, content);
        response.EnsureSuccessStatusCode();

        var jsonResponse = await response.Content.ReadAsStringAsync();
        var embeddingResponse = JsonSerializer.Deserialize<EmbeddingResponse>(jsonResponse);

        return embeddingResponse?.UserEmbedding ?? new List<float>();
    }
}
