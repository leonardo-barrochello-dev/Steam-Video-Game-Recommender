using SteamRecommenderAPI.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddOpenApi();

// Register HttpClient and Custom Services
builder.Services.AddHttpClient<SteamApiService>();
builder.Services.AddHttpClient<EmbeddingService>();
builder.Services.AddHttpClient<RecommendService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
