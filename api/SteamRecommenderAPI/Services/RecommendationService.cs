using System.Data;
using Microsoft.Data.Sqlite;

namespace SteamRecommenderAPI.Services;

public class GameRecommendation
{
    public int AppId { get; set; }
    public string Name { get; set; } = string.Empty;
    public float Score { get; set; }
}

public class RecommendationService
{
    private readonly string _connectionString;

    public RecommendationService()
    {
        _connectionString = $"Data Source={GetDatabasePath()}";
    }

    private static string GetDatabasePath()
    {
        var currentDir = new DirectoryInfo(AppDomain.CurrentDomain.BaseDirectory);
        while (currentDir != null)
        {
            var dbDir = Path.Combine(currentDir.FullName, "database");
            var dbFile = Path.Combine(dbDir, "recommender.db");
            if (Directory.Exists(dbDir) && File.Exists(dbFile))
            {
                return dbFile;
            }
            currentDir = currentDir.Parent;
        }

        // Fallback to absolute path on the workspace
        return @"C:\Projetos\steam-video-game-recommender\database\recommender.db";
    }

    public async Task<List<GameRecommendation>> GetRecommendationsAsync(List<float> userEmbedding, List<int> ownedAppIds, int topN = 10)
    {
        var recommendations = new List<GameRecommendation>();

        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        var command = connection.CreateCommand();
        command.CommandText = "SELECT app_id, name, embedding FROM game_embeddings";

        using var reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            var appId = reader.GetInt32(0);
            
            // Skip already owned games
            if (ownedAppIds.Contains(appId))
                continue;

            var name = reader.GetString(1);
            var embeddingBytes = (byte[])reader["embedding"];
            
            var itemEmbedding = new float[embeddingBytes.Length / 4];
            Buffer.BlockCopy(embeddingBytes, 0, itemEmbedding, 0, embeddingBytes.Length);

            // Calculate dot product (cosine similarity since vectors are L2 normalized)
            float score = 0;
            for (int i = 0; i < userEmbedding.Count && i < itemEmbedding.Length; i++)
            {
                score += userEmbedding[i] * itemEmbedding[i];
            }

            recommendations.Add(new GameRecommendation
            {
                AppId = appId,
                Name = name,
                Score = score
            });
        }

        var recommendationsOrdered = recommendations.OrderByDescending(x => x.Score).ToList();

        return recommendationsOrdered.Take(topN).ToList();
    }
}
