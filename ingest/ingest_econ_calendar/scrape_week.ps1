<#
.SYNOPSIS
    Scrapes ForexFactory economic calendar for a specific week using Chrome CDP.

.DESCRIPTION
    Requires Chrome running with: --remote-debugging-port=9222

.PARAMETER Week
    Week to scrape in format like "jul14.2025"

.EXAMPLE
    .\scrape_week.ps1 -Week "jul14.2025"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Week
)

$ErrorActionPreference = "Stop"
$CDP_URL = "http://localhost:9222"

# Get available pages
try {
    $pages = Invoke-RestMethod -Uri "$CDP_URL/json" -UseBasicParsing
} catch {
    Write-Error "Cannot connect to Chrome at $CDP_URL. Start Chrome with: --remote-debugging-port=9222"
    exit 1
}

# Find a suitable page
$target = $pages | Where-Object { $_.type -eq "page" -and $_.url -notlike "*extension*" } | Select-Object -First 1
if (-not $target) {
    Write-Error "No suitable browser page found"
    exit 1
}

$wsUrl = $target.webSocketDebuggerUrl
Write-Host "Connected to: $($target.title)" -ForegroundColor Green

# Create WebSocket connection
$ws = New-Object System.Net.WebSockets.ClientWebSocket
$ct = [System.Threading.CancellationToken]::None
$ws.ConnectAsync([Uri]$wsUrl, $ct).Wait()

function Send-CDPCommand {
    param([int]$Id, [string]$Method, [hashtable]$Params = @{})

    $msg = @{
        id = $Id
        method = $Method
        params = $Params
    } | ConvertTo-Json -Compress -Depth 10

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($msg)
    $segment = [System.ArraySegment[byte]]::new($bytes)
    $ws.SendAsync($segment, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $ct).Wait()

    # Receive response
    $buffer = New-Object byte[] 65536
    $result = ""
    do {
        $segment = [System.ArraySegment[byte]]::new($buffer)
        $recv = $ws.ReceiveAsync($segment, $ct).Result
        $result += [System.Text.Encoding]::UTF8.GetString($buffer, 0, $recv.Count)
    } while (-not $recv.EndOfMessage)

    return $result | ConvertFrom-Json
}

# Navigate to ForexFactory
$url = "https://www.forexfactory.com/calendar?week=$Week"
Write-Host "Navigating to $url..." -ForegroundColor Cyan

$null = Send-CDPCommand -Id 1 -Method "Page.navigate" -Params @{ url = $url }
Start-Sleep -Seconds 3

# Extract events using JavaScript
$js = @'
(() => {
    const rows = document.querySelectorAll('tr.calendar__row');
    const events = [];
    let currentDate = '';
    let currentTime = '';

    rows.forEach(row => {
        const dateCell = row.querySelector('.calendar__date');
        if (dateCell) {
            const dateSpan = dateCell.querySelector('span');
            if (dateSpan && dateSpan.textContent.trim()) {
                currentDate = dateSpan.textContent.trim();
            }
        }

        const timeCell = row.querySelector('.calendar__time');
        if (timeCell && timeCell.textContent.trim()) {
            currentTime = timeCell.textContent.trim();
        }

        const currency = row.querySelector('.calendar__currency')?.textContent?.trim();
        const eventTitle = row.querySelector('.calendar__event-title')?.textContent?.trim();
        const actual = row.querySelector('.calendar__actual')?.textContent?.trim();
        const forecast = row.querySelector('.calendar__forecast')?.textContent?.trim();
        const previous = row.querySelector('.calendar__previous')?.textContent?.trim();
        const impactSpan = row.querySelector('.calendar__impact span');
        const impact = impactSpan ? impactSpan.className : '';

        if (eventTitle) {
            events.push({
                date: currentDate,
                time: currentTime,
                currency: currency || '',
                event: eventTitle,
                actual: actual || '',
                forecast: forecast || '',
                previous: previous || '',
                impact: impact
            });
        }
    });

    return JSON.stringify(events);
})()
'@

Write-Host "Extracting events..." -ForegroundColor Cyan
$response = Send-CDPCommand -Id 2 -Method "Runtime.evaluate" -Params @{
    expression = $js
    returnByValue = $true
}

$ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, "", $ct).Wait()

# Output the events JSON
if ($response.result.result.value) {
    $response.result.result.value
} else {
    Write-Error "Failed to extract events"
    exit 1
}
