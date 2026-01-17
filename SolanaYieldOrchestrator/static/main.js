const API_BASE = '';
const UPDATE_INTERVAL = 5000;

let navChart = null;
let allocationChart = null;
let pnlBreakdownChart = null;
let targetActualChart = null;
let metricsHistory = [];
let currentPrices = {};
let selectedSide = 'buy';
let eventSource = null;
let sseConnected = false;
let enabledAssets = [];
let supportedAssets = [];
let assetInfo = {};
let assetsLoaded = false;

async function fetchJSON(endpoint) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

function formatCurrency(value, decimals = 2) {
    if (value === null || value === undefined) return '$0.00';
    const num = parseFloat(value);
    if (isNaN(num)) return '$0.00';
    return '$' + num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function formatPercent(value, decimals = 2) {
    if (value === null || value === undefined) return '0.00%';
    const num = parseFloat(value);
    if (isNaN(num)) return '0.00%';
    return num.toFixed(decimals) + '%';
}

function formatNumber(value, decimals = 4) {
    if (value === null || value === undefined) return '0';
    const num = parseFloat(value);
    if (isNaN(num)) return '0';
    return num.toFixed(decimals);
}

function setValueClass(element, value) {
    element.classList.remove('positive', 'negative');
    if (value > 0) element.classList.add('positive');
    else if (value < 0) element.classList.add('negative');
}

function initSSE() {
    if (eventSource) {
        eventSource.close();
    }
    
    try {
        eventSource = new EventSource('/events');
        
        eventSource.onopen = function() {
            sseConnected = true;
            updateSSEIndicator(true);
            console.log('SSE connection established');
        };
        
        eventSource.onerror = function(e) {
            sseConnected = false;
            updateSSEIndicator(false);
            console.log('SSE connection error, will retry...');
        };
        
        eventSource.addEventListener('connected', function(e) {
            sseConnected = true;
            updateSSEIndicator(true);
        });
        
        eventSource.addEventListener('prices', function(e) {
            try {
                const data = JSON.parse(e.data);
                for (const [symbol, price] of Object.entries(data)) {
                    currentPrices[symbol] = price;
                }
                updateTradePreview();
            } catch (err) {
                console.error('Error processing price event:', err);
            }
        });
        
        eventSource.addEventListener('metrics', function(e) {
            try {
                const data = JSON.parse(e.data);
                if (data.nav) {
                    document.getElementById('total-nav').textContent = formatCurrency(data.nav);
                }
                if (data.pnl !== undefined) {
                    const pnlEl = document.getElementById('total-pnl');
                    pnlEl.textContent = formatCurrency(data.pnl);
                    setValueClass(pnlEl, data.pnl);
                }
                if (data.pnl_pct !== undefined) {
                    const returnEl = document.getElementById('return-pct');
                    returnEl.textContent = formatPercent(data.pnl_pct);
                    setValueClass(returnEl, data.pnl_pct);
                }
            } catch (err) {
                console.error('Error processing metrics event:', err);
            }
        });
        
        eventSource.addEventListener('heartbeat', function(e) {
            // Keep-alive heartbeat
        });
        
    } catch (error) {
        console.error('Failed to initialize SSE:', error);
        sseConnected = false;
        updateSSEIndicator(false);
    }
}

function updateSSEIndicator(connected) {
    const sseDot = document.querySelector('.sse-dot');
    const sseText = document.getElementById('sse-text');
    
    if (sseDot && sseText) {
        if (connected) {
            sseDot.classList.add('connected');
            sseText.textContent = 'Live';
        } else {
            sseDot.classList.remove('connected');
            sseText.textContent = 'Offline';
        }
    }
}

async function updateStatus() {
    const data = await fetchJSON('/api/status');
    if (!data) return;
    
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');
    
    if (data.status === 'running') {
        statusDot.classList.add('connected');
        statusText.textContent = 'Connected';
    } else {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Disconnected';
    }
    
    if (data.last_update) {
        const lastUpdate = new Date(data.last_update * 1000);
        document.getElementById('last-update').textContent = `Last update: ${lastUpdate.toLocaleTimeString()}`;
    }
}

async function updatePrices() {
    const data = await fetchJSON('/api/prices');
    if (!data) return;
    
    const priceGrid = document.getElementById('price-grid');
    if (!priceGrid) return;
    
    // Use enabled assets from API response if available, otherwise use local state
    let symbols = enabledAssets;
    if (symbols.length === 0 && data.supported_assets) {
        symbols = data.supported_assets;
    }
    if (symbols.length === 0) {
        symbols = ['SOL', 'BTC', 'ETH', 'USDC', 'USDT'];
    }
    
    let html = '';
    for (const symbol of symbols) {
        const agg = data.aggregated?.[symbol];
        const coingecko = data.coingecko?.[symbol];
        const pyth = data.pyth?.[symbol];
        const venues = agg?.venues || {};
        
        const mainPrice = agg?.price || agg?.average_price || (symbol === 'USDC' || symbol === 'USDT' ? 1.0 : 0);
        
        currentPrices[symbol] = mainPrice;
        
        const change = coingecko?.change_24h || 0;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const changeSign = change >= 0 ? '+' : '';
        
        const isStablecoin = symbol === 'USDC' || symbol === 'USDT';
        const priceDisplay = isStablecoin ? `$${mainPrice.toFixed(4)}` : formatCurrency(mainPrice);
        
        const oracleStatus = agg?.oracle_status || 'unknown';
        const oracleDeviation = agg?.oracle_deviation_bps;
        const pythPrice = agg?.pyth_price;
        const crossVenue = agg?.cross_venue_tradable || false;
        
        const oracleBadge = data.pyth_enabled && oracleStatus !== 'unknown' 
            ? `<span class="oracle-badge ${oracleStatus}" title="Oracle deviation: ${oracleDeviation ? oracleDeviation.toFixed(1) + ' bps' : 'N/A'}">${oracleStatus}</span>` 
            : '';
        
        const crossVenueBadge = crossVenue ? '<span class="cross-venue-badge" title="Available on multiple perp venues">XV</span>' : '';
        
        const jAvail = venues.jupiter?.available;
        const cgAvail = venues.coingecko?.available;
        const kAvail = venues.kraken?.available;
        const hlAvail = venues.hyperliquid?.available;
        const drAvail = venues.drift?.available;
        const pythAvail = venues.pyth?.available;
        
        const venueBadgesHtml = `
            <div class="venue-badges">
                <span class="venue-badge jupiter ${jAvail ? 'available' : ''}" title="Jupiter${jAvail ? ': $' + venues.jupiter.price?.toFixed(4) : ' (N/A)'}">J</span>
                <span class="venue-badge coingecko ${cgAvail ? 'available' : ''}" title="CoinGecko${cgAvail ? ': $' + venues.coingecko.price?.toFixed(4) : ' (N/A)'}">CG</span>
                <span class="venue-badge kraken ${kAvail ? 'available' : ''}" title="Kraken${kAvail ? ': $' + venues.kraken.price?.toFixed(4) : ' (N/A)'}">K</span>
                <span class="venue-badge hyperliquid ${hlAvail ? 'available' : ''}" title="Hyperliquid${hlAvail ? ': $' + venues.hyperliquid.price?.toFixed(4) : ' (N/A)'}">HL</span>
                <span class="venue-badge drift ${drAvail ? 'available' : ''}" title="Drift${drAvail ? ': $' + venues.drift.mark_price?.toFixed(4) : ' (N/A)'}">DR</span>
                <span class="venue-badge pyth ${pythAvail ? 'available' : ''}" title="Pyth Oracle${pythAvail ? ': $' + venues.pyth.price?.toFixed(4) : ' (N/A)'}">PY</span>
            </div>
        `;
        
        let oracleInlineHtml = '';
        if (data.pyth_enabled && pythPrice && oracleStatus !== 'unknown') {
            oracleInlineHtml = `
                <div class="oracle-inline">
                    <span class="pyth-price">Pyth: $${pythPrice.toFixed(4)}</span>
                    <span class="deviation ${oracleStatus}">(${oracleDeviation?.toFixed(1) || '?'} bps)</span>
                </div>
            `;
        }
        
        html += `
            <div class="price-item ${isStablecoin ? 'stablecoin' : ''}">
                <div class="symbol">
                    <div class="symbol-icon ${symbol.toLowerCase()}"></div>
                    <span>${symbol}</span>
                    ${oracleBadge}
                    ${crossVenueBadge}
                </div>
                <div class="prices">
                    <div class="main-price">${priceDisplay} <span class="${changeClass}">${changeSign}${change.toFixed(2)}%</span></div>
                    ${venueBadgesHtml}
                    ${oracleInlineHtml}
                </div>
            </div>
        `;
    }
    
    priceGrid.innerHTML = html;
}

async function updateMetrics() {
    const data = await fetchJSON('/api/metrics');
    if (!data) return;
    
    const vault = data.vault || {};
    const portfolio = data.portfolio || {};
    const strategies = data.strategies || {};
    const allocations = data.allocations || {};
    const fundingRates = data.funding_rates || {};
    const risk = data.risk || {};
    const extendedMetrics = data.extended_metrics || {};
    const pnlBreakdown = data.pnl_breakdown || {};
    
    document.getElementById('total-nav').textContent = formatCurrency(vault.total_nav);
    
    const pnlEl = document.getElementById('total-pnl');
    pnlEl.textContent = formatCurrency(portfolio.total_pnl);
    setValueClass(pnlEl, portfolio.total_pnl);
    
    const returnEl = document.getElementById('return-pct');
    returnEl.textContent = formatPercent(portfolio.total_pnl_pct);
    setValueClass(returnEl, portfolio.total_pnl_pct);
    
    document.getElementById('max-drawdown').textContent = formatPercent(portfolio.max_drawdown || extendedMetrics.max_drawdown);
    
    document.getElementById('sharpe-ratio').textContent = formatNumber(extendedMetrics.sharpe_ratio || 0, 2);
    document.getElementById('sortino-ratio').textContent = formatNumber(extendedMetrics.sortino_ratio || 0, 2);
    document.getElementById('nav-volatility').textContent = formatPercent(extendedMetrics.nav_volatility || 0);
    
    metricsHistory.push({
        timestamp: Date.now(),
        nav: vault.total_nav || 0
    });
    
    if (metricsHistory.length > 500) {
        metricsHistory = metricsHistory.slice(-250);
    }
    
    updateNavChart();
    updateAllocationChart(allocations);
    updateFundingTable(fundingRates);
    updateStrategies(strategies);
    updateRiskIndicators(risk, portfolio);
    updatePnLBreakdown(pnlBreakdown);
}

function updatePnLBreakdown(breakdown) {
    const fundingEl = document.getElementById('pnl-funding');
    const basisEl = document.getElementById('pnl-basis');
    const feesEl = document.getElementById('pnl-fees');
    const mtmEl = document.getElementById('pnl-mtm');
    
    if (fundingEl) {
        fundingEl.textContent = formatCurrency(breakdown.funding || 0);
        setValueClass(fundingEl, breakdown.funding || 0);
    }
    if (basisEl) {
        basisEl.textContent = formatCurrency(breakdown.basis || 0);
        setValueClass(basisEl, breakdown.basis || 0);
    }
    if (feesEl) {
        feesEl.textContent = formatCurrency(-(breakdown.fees || 0));
        setValueClass(feesEl, -(breakdown.fees || 0));
    }
    if (mtmEl) {
        mtmEl.textContent = formatCurrency(breakdown.mark_to_market || 0);
        setValueClass(mtmEl, breakdown.mark_to_market || 0);
    }
    
    updatePnLBreakdownChart(breakdown);
}

function updatePnLBreakdownChart(breakdown) {
    const ctx = document.getElementById('pnl-breakdown-chart');
    if (!ctx) return;
    
    const data = [
        Math.max(0, breakdown.funding || 0),
        Math.max(0, breakdown.basis || 0),
        -(breakdown.fees || 0),
        breakdown.mark_to_market || 0
    ];
    
    const labels = ['Funding', 'Basis', 'Fees', 'Mark-to-Market'];
    const colors = ['#14F195', '#9945FF', '#FF6B35', '#3B82F6'];
    
    if (pnlBreakdownChart) {
        pnlBreakdownChart.data.datasets[0].data = data;
        pnlBreakdownChart.update('none');
    } else {
        pnlBreakdownChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 0,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#a0a0b0',
                            callback: (value) => '$' + value.toFixed(0)
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#a0a0b0' }
                    }
                }
            }
        });
    }
}

function updateNavChart() {
    const ctx = document.getElementById('nav-chart');
    if (!ctx) return;
    
    const labels = metricsHistory.map(m => new Date(m.timestamp).toLocaleTimeString());
    const navData = metricsHistory.map(m => m.nav);
    
    if (navChart) {
        navChart.data.labels = labels;
        navChart.data.datasets[0].data = navData;
        navChart.update('none');
    } else {
        navChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'NAV',
                    data: navData,
                    borderColor: '#14F195',
                    backgroundColor: 'rgba(20, 241, 149, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#a0a0b0',
                            callback: (value) => '$' + value.toLocaleString()
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }
}

function updateAllocationChart(allocations) {
    const ctx = document.getElementById('allocation-chart');
    if (!ctx) return;
    
    const actual = allocations.actual || {};
    const labels = Object.keys(actual);
    const data = Object.values(actual).map(v => (v * 100).toFixed(1));
    
    const colors = {
        'basis_harvester': '#9945FF',
        'funding_rotator': '#FF6B35',
        'cash': '#3B82F6'
    };
    
    const backgroundColors = labels.map(l => colors[l] || '#666');
    
    if (allocationChart) {
        allocationChart.data.labels = labels.map(l => l.replace('_', ' ').toUpperCase());
        allocationChart.data.datasets[0].data = data;
        allocationChart.data.datasets[0].backgroundColor = backgroundColors;
        allocationChart.update('none');
    } else {
        allocationChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels.map(l => l.replace('_', ' ').toUpperCase()),
                datasets: [{
                    data: data,
                    backgroundColor: backgroundColors,
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
    const legendEl = document.getElementById('allocation-legend');
    let legendHtml = '';
    labels.forEach((label, i) => {
        legendHtml += `
            <div class="legend-item">
                <div class="legend-dot" style="background: ${backgroundColors[i]}"></div>
                <span>${label.replace('_', ' ')}: ${data[i]}%</span>
            </div>
        `;
    });
    legendEl.innerHTML = legendHtml;
}

function updateFundingTable(fundingRates) {
    const table = document.getElementById('funding-table');
    if (!table) return;
    
    let html = `
        <div class="funding-row header">
            <span class="market">Market</span>
            <span class="rate">Rate (1H)</span>
            <span class="apy">APY</span>
        </div>
    `;
    
    const markets = Object.entries(fundingRates).sort((a, b) => {
        const apyA = Math.abs(a[1]?.apy || 0);
        const apyB = Math.abs(b[1]?.apy || 0);
        return apyB - apyA;
    });
    
    for (const [market, data] of markets.slice(0, 5)) {
        const rate = data?.rate || 0;
        const apy = data?.apy || 0;
        const apyClass = apy >= 0 ? 'positive' : 'negative';
        
        html += `
            <div class="funding-row">
                <span class="market">${market}</span>
                <span class="rate">${(rate * 100).toFixed(4)}%</span>
                <span class="apy ${apyClass}">${apy.toFixed(2)}%</span>
            </div>
        `;
    }
    
    table.innerHTML = html;
}

function updateStrategies(strategies) {
    const basis = strategies.basis_harvester || {};
    document.getElementById('basis-current').textContent = `${(basis.current_basis_bps || 0).toFixed(1)} bps`;
    document.getElementById('basis-apy').textContent = formatPercent(basis.current_basis_apy);
    document.getElementById('basis-positions').textContent = basis.active_positions || 0;
    
    const basisPnl = (basis.unrealized_pnl || 0) + (basis.realized_pnl || 0);
    const basisPnlEl = document.getElementById('basis-pnl');
    basisPnlEl.textContent = formatCurrency(basisPnl);
    setValueClass(basisPnlEl, basisPnl);
    
    const funding = strategies.funding_rotator || {};
    const topMarkets = funding.top_funding_markets || [];
    document.getElementById('funding-top-market').textContent = topMarkets[0]?.market || '-';
    document.getElementById('funding-best-apy').textContent = formatPercent(topMarkets[0]?.funding_apy);
    document.getElementById('funding-positions').textContent = funding.active_positions || 0;
    
    const fundingEarned = funding.total_funding_earned || 0;
    const fundingEarnedEl = document.getElementById('funding-earned');
    fundingEarnedEl.textContent = formatCurrency(fundingEarned);
    setValueClass(fundingEarnedEl, fundingEarned);
}

function updateRiskIndicators(risk, portfolio) {
    const riskLevel = document.getElementById('risk-level');
    const level = (risk.overall_risk_level || 'low').toUpperCase();
    riskLevel.textContent = level;
    riskLevel.className = `risk-badge ${level.toLowerCase()}`;
    
    const drawdown = portfolio.max_drawdown || 0;
    const maxDrawdown = 10;
    const drawdownPct = Math.min((drawdown / maxDrawdown) * 100, 100);
    document.getElementById('drawdown-progress').style.width = `${drawdownPct}%`;
    
    document.getElementById('leverage-value').textContent = '1.0x';
}

async function runSimulation() {
    const btn = document.getElementById('run-simulation');
    const resultsEl = document.getElementById('simulator-results');
    
    btn.disabled = true;
    btn.textContent = 'Running...';
    resultsEl.innerHTML = '<p class="hint">Running simulation...</p>';
    
    const steps = parseInt(document.getElementById('sim-steps').value) || 50;
    const capital = parseFloat(document.getElementById('sim-capital').value) || 10000;
    
    try {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps, capital })
        });
        
        const data = await response.json();
        
        if (data.success && data.results) {
            const r = data.results;
            resultsEl.innerHTML = `
                <div class="sim-results-grid">
                    <div class="sim-result-item">
                        <span>Final NAV</span>
                        <span>${formatCurrency(r.final_nav)}</span>
                    </div>
                    <div class="sim-result-item">
                        <span>Total Return</span>
                        <span class="${r.total_return_pct >= 0 ? 'positive' : 'negative'}">${formatPercent(r.total_return_pct)}</span>
                    </div>
                    <div class="sim-result-item">
                        <span>Max Drawdown</span>
                        <span>${formatPercent(r.max_drawdown_pct)}</span>
                    </div>
                    <div class="sim-result-item">
                        <span>Sharpe Ratio</span>
                        <span>${r.sharpe_ratio?.toFixed(2) || '0.00'}</span>
                    </div>
                    <div class="sim-result-item">
                        <span>Volatility</span>
                        <span>${formatPercent(r.volatility_pct)}</span>
                    </div>
                    <div class="sim-result-item">
                        <span>Total Trades</span>
                        <span>${r.total_trades || 0}</span>
                    </div>
                </div>
            `;
        } else {
            resultsEl.innerHTML = `<p class="hint">Error: ${data.error || 'Unknown error'}</p>`;
        }
    } catch (error) {
        resultsEl.innerHTML = `<p class="hint">Error: ${error.message}</p>`;
    }
    
    btn.disabled = false;
    btn.textContent = 'Run Simulation';
}

async function runScenario() {
    const btn = document.getElementById('run-scenario');
    const resultsEl = document.getElementById('scenario-results');
    
    btn.disabled = true;
    btn.textContent = 'Running...';
    resultsEl.innerHTML = '<p class="hint">Running scenario analysis...</p>';
    
    const solShock = parseFloat(document.getElementById('scenario-sol-shock').value) || 0;
    const btcShock = parseFloat(document.getElementById('scenario-btc-shock').value) || 0;
    
    try {
        const response = await fetch('/api/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                price_shocks: {
                    SOL: solShock,
                    BTC: btcShock
                }
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.scenario) {
            const s = data.scenario;
            const navChangeClass = s.nav_change_pct >= 0 ? 'positive' : 'negative';
            const liqWarnings = s.liq_warnings || [];
            
            let warningsHtml = '';
            if (liqWarnings.length > 0) {
                warningsHtml = `
                    <div class="scenario-warnings">
                        <strong>Liquidation Warnings:</strong>
                        ${liqWarnings.map(w => `<div class="warning-item ${w.severity}">${w.market}: ${formatPercent(w.distance_to_liq_pct)} from liq</div>`).join('')}
                    </div>
                `;
            }
            
            resultsEl.innerHTML = `
                <div class="scenario-summary">
                    <div class="scenario-nav">
                        <div class="scenario-nav-item">
                            <span>Current NAV</span>
                            <span>${formatCurrency(s.current_nav)}</span>
                        </div>
                        <div class="scenario-arrow">→</div>
                        <div class="scenario-nav-item">
                            <span>Shocked NAV</span>
                            <span class="${navChangeClass}">${formatCurrency(s.shocked_nav)}</span>
                        </div>
                    </div>
                    <div class="scenario-change ${navChangeClass}">
                        NAV Change: ${formatCurrency(s.nav_change)} (${s.nav_change_pct >= 0 ? '+' : ''}${formatPercent(s.nav_change_pct)})
                    </div>
                    ${warningsHtml}
                </div>
            `;
        } else {
            resultsEl.innerHTML = `<p class="hint">Error: ${data.error || 'Unknown error'}</p>`;
        }
    } catch (error) {
        resultsEl.innerHTML = `<p class="hint">Error: ${error.message}</p>`;
    }
    
    btn.disabled = false;
    btn.textContent = 'Run Scenario';
}

function initChartControls() {
    const buttons = document.querySelectorAll('.chart-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

function initTradeTicket() {
    const venueSelect = document.getElementById('trade-venue');
    const assetSelect = document.getElementById('trade-asset');
    const leverageGroup = document.getElementById('leverage-group');
    const sideBuy = document.getElementById('side-buy');
    const sideSell = document.getElementById('side-sell');
    const sizeInput = document.getElementById('trade-size');
    const submitBtn = document.getElementById('submit-trade');
    
    venueSelect.addEventListener('change', () => {
        const isPerp = venueSelect.value === 'drift';
        leverageGroup.style.display = isPerp ? 'block' : 'none';
        updateTradePreview();
    });
    
    sideBuy.addEventListener('click', () => {
        selectedSide = venueSelect.value === 'drift' ? 'long' : 'buy';
        sideBuy.classList.add('active');
        sideSell.classList.remove('active');
    });
    
    sideSell.addEventListener('click', () => {
        selectedSide = venueSelect.value === 'drift' ? 'short' : 'sell';
        sideSell.classList.add('active');
        sideBuy.classList.remove('active');
    });
    
    sizeInput.addEventListener('input', updateTradePreview);
    assetSelect.addEventListener('change', updateTradePreview);
    
    submitBtn.addEventListener('click', submitTrade);
}

function updateTradePreview() {
    const symbol = document.getElementById('trade-asset')?.value || 'SOL';
    const venue = document.getElementById('trade-venue')?.value || 'jupiter';
    const size = parseFloat(document.getElementById('trade-size').value) || 0;
    
    let price = currentPrices[symbol] || 100;
    
    const cost = size * price;
    const marketLabel = venue === 'drift' ? `${symbol}-PERP` : `${symbol}/USDC`;
    
    document.getElementById('preview-price').textContent = formatCurrency(price);
    document.getElementById('preview-cost').textContent = formatCurrency(cost);
}

async function submitTrade() {
    const btn = document.getElementById('submit-trade');
    const resultDiv = document.getElementById('trade-result');
    
    const venue = document.getElementById('trade-venue').value;
    const symbol = document.getElementById('trade-asset').value;
    const market = venue === 'drift' ? `${symbol}-PERP` : `${symbol}/USDC`;
    const size = parseFloat(document.getElementById('trade-size').value) || 0;
    const slippage = parseInt(document.getElementById('trade-slippage').value);
    const priority = document.getElementById('trade-priority').value;
    const leverage = parseFloat(document.getElementById('trade-leverage').value) || 1;
    
    btn.disabled = true;
    btn.textContent = 'Executing...';
    
    try {
        const response = await fetch('/api/trade/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                venue,
                market,
                side: selectedSide,
                size,
                slippage_bps: slippage,
                priority_profile: priority,
                leverage
            })
        });
        
        const data = await response.json();
        
        resultDiv.style.display = 'block';
        
        if (data.success) {
            resultDiv.innerHTML = `
                <div class="trade-success">
                    <span class="trade-icon">&#10003;</span>
                    <div>
                        <strong>${selectedSide.toUpperCase()} ${size} ${market}</strong>
                        <span>@ ${formatCurrency(data.filled_price)}</span>
                    </div>
                </div>
            `;
            updatePositions();
            updatePerpRisk();
        } else {
            resultDiv.innerHTML = `
                <div class="trade-error">
                    <span class="trade-icon">&#10007;</span>
                    <span>${data.error || 'Trade failed'}</span>
                </div>
            `;
        }
        
        setTimeout(() => {
            resultDiv.style.display = 'none';
        }, 5000);
        
    } catch (error) {
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `<div class="trade-error">${error.message}</div>`;
    }
    
    btn.disabled = false;
    btn.textContent = 'Execute Trade';
}

async function updatePositions() {
    const data = await fetchJSON('/api/user/positions');
    if (!data || !data.success) return;
    
    const tableEl = document.getElementById('positions-table');
    const positions = data.positions || { spot: [], perp: [] };
    
    let html = `
        <div class="positions-row header">
            <span class="col-market">Market</span>
            <span class="col-side">Side</span>
            <span class="col-size">Size</span>
            <span class="col-entry">Entry</span>
            <span class="col-current">Current</span>
            <span class="col-pnl">PnL</span>
            <span class="col-liq">Liq. Price</span>
        </div>
    `;
    
    const allPositions = [...(positions.spot || []), ...(positions.perp || [])];
    
    if (allPositions.length === 0) {
        html += '<div class="positions-empty">No open positions</div>';
    } else {
        for (const pos of allPositions) {
            const pnlClass = pos.unrealized_pnl >= 0 ? 'positive' : 'negative';
            const sideClass = ['buy', 'long'].includes(pos.side?.toLowerCase()) ? 'buy' : 'sell';
            
            html += `
                <div class="positions-row">
                    <span class="col-market">${pos.market || '-'}</span>
                    <span class="col-side ${sideClass}">${(pos.side || '-').toUpperCase()}</span>
                    <span class="col-size">${formatNumber(pos.size, 4)}</span>
                    <span class="col-entry">${formatCurrency(pos.avg_entry_price || pos.entry_price)}</span>
                    <span class="col-current">${formatCurrency(pos.current_price)}</span>
                    <span class="col-pnl ${pnlClass}">${formatCurrency(pos.unrealized_pnl)}</span>
                    <span class="col-liq">${pos.liquidation_price ? formatCurrency(pos.liquidation_price) : '-'}</span>
                </div>
            `;
        }
    }
    
    tableEl.innerHTML = html;
    
    const account = data.account || {};
    const navEl = document.getElementById('total-nav');
    if (navEl && account.current_nav) {
        navEl.textContent = formatCurrency(account.current_nav);
    }
}

async function updatePerpRisk() {
    const data = await fetchJSON('/api/perp/risk');
    if (!data || !data.success) return;
    
    const metrics = data.risk_metrics?.aggregate || {};
    
    document.getElementById('perp-notional').textContent = formatCurrency(metrics.total_notional);
    document.getElementById('perp-margin').textContent = formatCurrency(metrics.total_margin);
    document.getElementById('perp-avg-leverage').textContent = `${(metrics.weighted_avg_leverage || 0).toFixed(1)}x`;
    
    const riskLevel = document.getElementById('perp-risk-level');
    const level = (metrics.overall_risk || 'low').toUpperCase();
    riskLevel.textContent = level;
    riskLevel.className = `risk-badge ${level.toLowerCase()}`;
    
    const warningsEl = document.getElementById('perp-warnings');
    const warnings = data.risk_metrics?.warnings || [];
    
    if (warnings.length > 0) {
        let warningsHtml = '';
        for (const w of warnings) {
            warningsHtml += `
                <div class="warning-item ${w.severity}">
                    <span class="warning-icon">&#9888;</span>
                    <span>${w.market}: ${w.type.replace('_', ' ')}</span>
                </div>
            `;
        }
        warningsEl.innerHTML = warningsHtml;
    } else {
        warningsEl.innerHTML = '';
    }
}

function initSimSettings() {
    const saveBtn = document.getElementById('save-sim-config');
    const resetBtn = document.getElementById('reset-sim');
    const profileSelect = document.getElementById('strategy-profile');
    
    loadSimConfig();
    
    profileSelect.addEventListener('change', onProfileChange);
    saveBtn.addEventListener('click', saveSimConfig);
    resetBtn.addEventListener('click', resetSimAccount);
    
    document.getElementById('run-scenario').addEventListener('click', runScenario);
}

function onProfileChange() {
    const profile = document.getElementById('strategy-profile').value;
    const descEl = document.getElementById('profile-description');
    
    const descriptions = {
        'conservative': 'Low risk, focus on stable yield with minimal drawdown. Lower leverage and position sizes.',
        'balanced': 'Moderate risk with balanced yield and risk exposure. Standard settings.',
        'aggro': 'High risk, maximum yield pursuit with higher leverage. Aggressive entry thresholds.'
    };
    
    if (profile && descriptions[profile]) {
        descEl.textContent = descriptions[profile];
        descEl.style.display = 'block';
    } else {
        descEl.style.display = 'none';
    }
}

async function loadSimConfig() {
    const data = await fetchJSON('/api/sim/config');
    if (!data || !data.success) return;
    
    const config = data.config || {};
    
    document.getElementById('sim-nav').value = config.initial_nav || 10000;
    
    if (config.profile) {
        document.getElementById('strategy-profile').value = config.profile;
        onProfileChange();
    }
    
    const strategies = config.strategies || {};
    document.getElementById('strat-basis').checked = strategies.basis !== false;
    document.getElementById('strat-funding').checked = strategies.funding !== false;
    document.getElementById('strat-carry').checked = strategies.carry === true;
    document.getElementById('strat-volatility').checked = strategies.volatility === true;
    document.getElementById('strat-basket').checked = strategies.basket === true;
    
    const thresholds = config.thresholds || {};
    document.getElementById('sim-basis-entry').value = thresholds.basis_entry_bps || 50;
    document.getElementById('sim-funding-min').value = thresholds.funding_min_apy || 10;
}

async function saveSimConfig() {
    const btn = document.getElementById('save-sim-config');
    btn.disabled = true;
    btn.textContent = 'Saving...';
    
    const profile = document.getElementById('strategy-profile').value;
    
    const payload = {
        initial_nav: parseFloat(document.getElementById('sim-nav').value) || 10000,
        strategies: {
            basis: document.getElementById('strat-basis').checked,
            funding: document.getElementById('strat-funding').checked,
            carry: document.getElementById('strat-carry').checked,
            volatility: document.getElementById('strat-volatility').checked,
            basket: document.getElementById('strat-basket').checked
        },
        thresholds: {
            basis_entry_bps: parseFloat(document.getElementById('sim-basis-entry').value) || 50,
            funding_min_apy: parseFloat(document.getElementById('sim-funding-min').value) || 10
        }
    };
    
    if (profile) {
        payload.profile = profile;
    }
    
    try {
        const response = await fetch('/api/sim/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadSimConfig();
            updateMetrics();
        } else {
            alert('Error saving config: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error saving config: ' + error.message);
    }
    
    btn.disabled = false;
    btn.textContent = 'Apply Settings';
}

async function resetSimAccount() {
    const btn = document.getElementById('reset-sim');
    
    if (!confirm('Reset paper account? This will clear all positions and PnL.')) {
        return;
    }
    
    btn.disabled = true;
    btn.textContent = 'Resetting...';
    
    try {
        const response = await fetch('/api/sim/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                initial_nav: parseFloat(document.getElementById('sim-nav').value) || 10000
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            metricsHistory = [];
            updateMetrics();
            updatePositions();
            updatePerpRisk();
        } else {
            alert('Error resetting account: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error resetting account: ' + error.message);
    }
    
    btn.disabled = false;
    btn.textContent = 'Reset Account';
}

function initPositionsTabs() {
    const tabs = document.querySelectorAll('.positions-tabs .tab-btn');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
        });
    });
}

function initPortfolioBuilder() {
    const weightSliders = document.querySelectorAll('.weight-slider');
    const saveBtn = document.getElementById('save-portfolio-config');
    
    weightSliders.forEach(slider => {
        slider.addEventListener('input', updateWeightSliders);
    });
    
    if (saveBtn) {
        saveBtn.addEventListener('click', savePortfolioConfig);
    }
    
    loadPortfolioConfig();
    updatePortfolioState();
}

function updateWeightSliders() {
    const basis = parseInt(document.getElementById('weight-basis').value) || 0;
    const funding = parseInt(document.getElementById('weight-funding').value) || 0;
    const cash = parseInt(document.getElementById('weight-cash').value) || 0;
    
    document.getElementById('weight-basis-value').textContent = basis + '%';
    document.getElementById('weight-funding-value').textContent = funding + '%';
    document.getElementById('weight-cash-value').textContent = cash + '%';
    
    const total = basis + funding + cash;
    const totalEl = document.getElementById('weight-total');
    totalEl.textContent = `Total: ${total}%`;
    totalEl.className = 'weight-total' + (total === 100 ? '' : ' warning');
}

function updateAssetSliders() {
    const container = document.getElementById('portfolio-sliders');
    if (!container) return;
    
    let total = 0;
    container.querySelectorAll('.portfolio-slider').forEach(slider => {
        const symbol = slider.dataset.symbol;
        const value = parseInt(slider.value) || 0;
        total += value;
        
        const valueEl = document.getElementById(`slider-${symbol}-value`);
        if (valueEl) valueEl.textContent = `${value}%`;
    });
    
    const totalEl = document.getElementById('asset-total');
    if (totalEl) {
        totalEl.textContent = `Total: ${total}%`;
        totalEl.className = 'asset-total' + (total === 100 ? '' : ' warning');
    }
}

async function loadPortfolioConfig() {
    const data = await fetchJSON('/api/portfolio/config');
    if (!data || !data.success) return;
    
    const navEl = document.getElementById('portfolio-nav');
    if (navEl) navEl.value = data.initial_nav || 10000;
    
    const stratWeights = data.strategy_weights || {};
    const basisPct = Math.round((stratWeights.basis || 0.5) * 100);
    const fundingPct = Math.round((stratWeights.funding || 0.3) * 100);
    const cashPct = Math.round((stratWeights.cash || 0.2) * 100);
    
    const basisEl = document.getElementById('weight-basis');
    const fundingEl = document.getElementById('weight-funding');
    const cashEl = document.getElementById('weight-cash');
    
    if (basisEl) basisEl.value = basisPct;
    if (fundingEl) fundingEl.value = fundingPct;
    if (cashEl) cashEl.value = cashPct;
    
    const assetWeights = data.asset_weights || {};
    const container = document.getElementById('portfolio-sliders');
    if (container) {
        container.querySelectorAll('.portfolio-slider').forEach(slider => {
            const symbol = slider.dataset.symbol;
            if (symbol && assetWeights[symbol] !== undefined) {
                slider.value = Math.round(assetWeights[symbol] * 100);
            }
        });
    }
    
    updateWeightSliders();
    updateAssetSliders();
}

async function savePortfolioConfig() {
    const btn = document.getElementById('save-portfolio-config');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Saving...';
    }
    
    const basis = parseInt(document.getElementById('weight-basis')?.value || 50) / 100;
    const funding = parseInt(document.getElementById('weight-funding')?.value || 30) / 100;
    const cash = parseInt(document.getElementById('weight-cash')?.value || 20) / 100;
    
    const assetWeights = {};
    const container = document.getElementById('portfolio-sliders');
    if (container) {
        container.querySelectorAll('.portfolio-slider').forEach(slider => {
            const symbol = slider.dataset.symbol;
            if (symbol) {
                assetWeights[symbol] = parseInt(slider.value || 0) / 100;
            }
        });
    }
    
    const payload = {
        initial_nav: parseFloat(document.getElementById('portfolio-nav')?.value || 10000),
        strategy_weights: { basis, funding, cash },
        asset_weights: assetWeights
    };
    
    try {
        const response = await fetch('/api/portfolio/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.success) {
            updatePortfolioState();
            updateMetrics();
        } else {
            alert('Error saving portfolio config: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error saving portfolio config: ' + error.message);
    }
    
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Save Configuration';
    }
}

async function updatePortfolioState() {
    const data = await fetchJSON('/api/portfolio/state');
    if (!data || !data.success) return;
    
    document.getElementById('portfolio-current-nav').textContent = formatCurrency(data.nav);
    
    const pnlEl = document.getElementById('portfolio-pnl');
    pnlEl.textContent = formatCurrency(data.pnl);
    setValueClass(pnlEl, data.pnl);
    
    const returnEl = document.getElementById('portfolio-return');
    returnEl.textContent = formatPercent(data.pnl_pct);
    setValueClass(returnEl, data.pnl_pct);
    
    updateTargetActualChart(data);
}

function updateTargetActualChart(data) {
    const ctx = document.getElementById('target-actual-chart');
    if (!ctx) return;
    
    const target = data.target_strategy_weights || {};
    const actual = data.actual_strategy_weights || {};
    
    const labels = ['Basis', 'Funding', 'Cash'];
    const targetData = [
        (target.basis || 0) * 100,
        (target.funding || 0) * 100,
        (target.cash || 0) * 100
    ];
    const actualData = [
        (actual.basis || 0) * 100,
        (actual.funding || 0) * 100,
        (actual.cash || 0) * 100
    ];
    
    if (targetActualChart) {
        targetActualChart.data.datasets[0].data = targetData;
        targetActualChart.data.datasets[1].data = actualData;
        targetActualChart.update('none');
    } else {
        targetActualChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Target',
                        data: targetData,
                        backgroundColor: 'rgba(153, 69, 255, 0.7)',
                        borderColor: '#9945FF',
                        borderWidth: 1
                    },
                    {
                        label: 'Actual',
                        data: actualData,
                        backgroundColor: 'rgba(20, 241, 149, 0.7)',
                        borderColor: '#14F195',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { color: '#a0a0b0' }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#a0a0b0' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: {
                            color: '#a0a0b0',
                            callback: (value) => value + '%'
                        },
                        max: 100
                    }
                }
            }
        });
    }
}

async function updateChainState() {
    const data = await fetchJSON('/api/chain/state');
    if (!data || !data.success) return;
    
    const slotEl = document.getElementById('chain-slot');
    const lastUpdateEl = document.getElementById('chain-last-update');
    const connectedEl = document.getElementById('chain-connected');
    
    if (slotEl) slotEl.textContent = data.latest_slot ? data.latest_slot.toLocaleString() : '-';
    if (lastUpdateEl && data.seconds_since_update !== null) {
        lastUpdateEl.textContent = `${Math.round(data.seconds_since_update)}s ago`;
    }
    if (connectedEl) connectedEl.textContent = data.is_connected ? 'Yes' : 'No';
}

async function updatePriorityFees() {
    const data = await fetchJSON('/api/chain/fees');
    if (!data || !data.success) return;
    
    const fees = data.priority_fees || {};
    const cheapEl = document.getElementById('fee-cheap');
    const balancedEl = document.getElementById('fee-balanced');
    const fastEl = document.getElementById('fee-fast');
    
    if (cheapEl) cheapEl.textContent = (fees.cheap || 0).toLocaleString();
    if (balancedEl) balancedEl.textContent = (fees.balanced || 0).toLocaleString();
    if (fastEl) fastEl.textContent = (fees.fast || 0).toLocaleString();
}

async function updateFundingTermStructure() {
    const data = await fetchJSON('/api/drift/funding/term-structure');
    if (!data || !data.success) return;
    
    const tableEl = document.getElementById('term-structure-table');
    if (!tableEl) return;
    
    const sortedByApy = data.sorted_by_apy || [];
    
    if (sortedByApy.length === 0) {
        tableEl.innerHTML = '<p class="hint">No funding data available</p>';
        return;
    }
    
    let html = `
        <div class="term-row header">
            <span>Market</span>
            <span>Hourly</span>
            <span>Daily</span>
            <span>APY</span>
            <span>Direction</span>
        </div>
    `;
    
    for (const market of sortedByApy.slice(0, 5)) {
        const apyClass = market.apy_pct > 0 ? 'positive' : market.apy_pct < 0 ? 'negative' : '';
        html += `
            <div class="term-row">
                <span class="symbol">${market.symbol}</span>
                <span>${market.hourly_rate.toFixed(2)} bps</span>
                <span>${market.daily_rate.toFixed(2)} bps</span>
                <span class="${apyClass}">${market.apy_pct.toFixed(1)}%</span>
                <span class="direction ${market.direction}">${market.direction.replace('_', ' ')}</span>
            </div>
        `;
    }
    
    tableEl.innerHTML = html;
}

async function updateMarginHeatmap() {
    const data = await fetchJSON('/api/drift/margin/heatmap');
    if (!data || !data.success) return;
    
    const totalEl = document.getElementById('margin-total');
    const utilEl = document.getElementById('margin-utilization');
    const positionsEl = document.getElementById('margin-positions');
    
    if (totalEl) totalEl.textContent = formatCurrency(data.total_margin_used);
    if (utilEl) utilEl.textContent = formatPercent(data.margin_utilization_pct);
    
    if (!positionsEl) return;
    
    const positions = data.positions || [];
    if (positions.length === 0) {
        positionsEl.innerHTML = '<p class="hint">No margin positions</p>';
        return;
    }
    
    let html = '';
    for (const pos of positions) {
        const ratioClass = pos.margin_ratio < 0.5 ? 'safe' : pos.margin_ratio < 0.8 ? 'warning' : 'danger';
        html += `
            <div class="margin-position ${ratioClass}">
                <span class="symbol">${pos.symbol}</span>
                <span class="leverage">${pos.leverage.toFixed(1)}x</span>
                <span class="ratio">${(pos.margin_ratio * 100).toFixed(0)}%</span>
            </div>
        `;
    }
    positionsEl.innerHTML = html;
}

async function updateHFTLatency() {
    const data = await fetchJSON('/api/hft/latency');
    if (!data || !data.success) return;
    
    const venues = data.venues || {};
    
    for (const [venue, metrics] of Object.entries(venues)) {
        const el = document.getElementById(`latency-${venue.toLowerCase()}`);
        if (el) {
            const latency = metrics.avg_latency_ms || 0;
            const callsPerMin = metrics.calls_last_minute || 0;
            const rateLimitUtil = (metrics.rate_limit_utilization || 0) * 100;
            const rateLimitStatus = metrics.rate_limit_status || 'ok';
            
            let displayText = `${latency.toFixed(0)}ms`;
            if (callsPerMin > 0) {
                displayText += ` | ${callsPerMin}/min`;
            }
            if (rateLimitUtil > 50) {
                displayText += ` | ${rateLimitUtil.toFixed(0)}%`;
            }
            
            el.textContent = displayText;
            
            let className = 'latency ';
            if (rateLimitStatus === 'near_limit') {
                className += 'slow';
            } else if (latency < 200) {
                className += 'good';
            } else if (latency < 500) {
                className += 'moderate';
            } else {
                className += 'slow';
            }
            el.className = className;
        }
    }
    
    const latencyGrid = document.getElementById('latency-grid');
    if (latencyGrid && Object.keys(venues).length > 3) {
        let html = '';
        const displayVenues = ['jupiter', 'drift', 'hyperliquid', 'coingecko', 'kraken', 'pyth'];
        
        for (const venue of displayVenues) {
            const metrics = venues[venue];
            if (!metrics) continue;
            
            const latency = metrics.avg_latency_ms || 0;
            const callsPerMin = metrics.calls_last_minute || 0;
            const rateLimitUtil = (metrics.rate_limit_utilization || 0) * 100;
            const rateLimitStatus = metrics.rate_limit_status || 'ok';
            
            const latencyClass = rateLimitStatus === 'near_limit' ? 'slow' : 
                                 latency < 200 ? 'good' : latency < 500 ? 'moderate' : 'slow';
            
            html += `
                <div class="latency-item">
                    <span class="venue">${venue.charAt(0).toUpperCase() + venue.slice(1)}</span>
                    <span class="latency ${latencyClass}">
                        ${latency.toFixed(0)}ms
                        ${callsPerMin > 0 ? `<span class="calls">${callsPerMin}/m</span>` : ''}
                        ${rateLimitUtil > 50 ? `<span class="rate-limit ${rateLimitStatus}">${rateLimitUtil.toFixed(0)}%</span>` : ''}
                    </span>
                </div>
            `;
        }
        
        if (html) {
            latencyGrid.innerHTML = html;
        }
    }
}

async function updateCrossVenueFunding() {
    const symbolsParam = enabledAssets.filter(s => s !== 'USDC' && s !== 'USDT').slice(0, 5).join(',');
    const data = await fetchJSON(`/api/cross-venue/funding?symbols=${symbolsParam || 'SOL,BTC,ETH'}`);
    if (!data || !data.success) return;
    
    const tableEl = document.getElementById('cross-venue-table');
    if (!tableEl) return;
    
    const crossVenueData = data.cross_venue_funding || {};
    
    let html = `
        <div class="cross-venue-row header">
            <span class="col-symbol">Symbol</span>
            <span class="col-drift">Drift</span>
            <span class="col-hl">Hyperliquid</span>
            <span class="col-diff">Diff</span>
            <span class="col-suggestion">Suggested</span>
        </div>
    `;
    
    for (const [symbol, info] of Object.entries(crossVenueData)) {
        const venues = info.venues || {};
        const driftRate = venues.drift ? `${(venues.drift.rate * 10000).toFixed(2)} bps` : '-';
        const hlRate = venues.hyperliquid ? `${(venues.hyperliquid.rate * 10000).toFixed(2)} bps` : '-';
        const diffBps = info.max_diff_bps || 0;
        const diffClass = diffBps >= 5 ? 'positive' : '';
        const suggested = info.suggested || '-';
        
        html += `
            <div class="cross-venue-row">
                <span class="col-symbol">${symbol}</span>
                <span class="col-drift">${driftRate}</span>
                <span class="col-hl">${hlRate}</span>
                <span class="col-diff ${diffClass}">${diffBps.toFixed(1)} bps</span>
                <span class="col-suggestion">${suggested}</span>
            </div>
        `;
    }
    
    tableEl.innerHTML = html;
}

async function fetchJupiterRoutes() {
    const inputToken = document.getElementById('jupiter-input-token').value;
    const outputToken = document.getElementById('jupiter-output-token').value;
    const amount = parseFloat(document.getElementById('jupiter-amount').value) || 1;
    
    const decimals = inputToken === 'SOL' ? 9 : 6;
    const amountRaw = Math.floor(amount * Math.pow(10, decimals));
    
    const resultsEl = document.getElementById('jupiter-routes-results');
    resultsEl.innerHTML = '<p class="loading">Fetching routes...</p>';
    
    const data = await fetchJSON(`/api/jupiter/routes/compare?input_token=${inputToken}&output_token=${outputToken}&amount=${amountRaw}`);
    
    if (!data || !data.success) {
        resultsEl.innerHTML = '<p class="error">Failed to fetch routes</p>';
        return;
    }
    
    const routes = data.routes || [];
    if (routes.length === 0) {
        resultsEl.innerHTML = '<p class="hint">No routes found</p>';
        return;
    }
    
    let html = '<div class="routes-list">';
    for (const route of routes) {
        const impactClass = route.price_impact_bps < 10 ? 'low' : route.price_impact_bps < 50 ? 'medium' : 'high';
        html += `
            <div class="route-item">
                <div class="route-path">${route.route || 'Direct'}</div>
                <div class="route-metrics">
                    <span>Slippage: ${route.slippage_setting} bps</span>
                    <span>Impact: <span class="${impactClass}">${route.price_impact_bps.toFixed(1)} bps</span></span>
                    <span>Hops: ${route.hops}</span>
                    <span>Output: ${(route.output_amount / 1e6).toFixed(2)}</span>
                </div>
            </div>
        `;
    }
    html += '</div>';
    
    resultsEl.innerHTML = html;
    
    const healthData = await fetchJSON('/api/jupiter/health');
    if (healthData && healthData.success) {
        const healthEl = document.getElementById('jupiter-health');
        const status = healthData.jupiter.status || 'unknown';
        const statusClass = status === 'good' ? 'good' : status === 'degraded' ? 'warning' : 'error';
        healthEl.innerHTML = `<span class="health-badge ${statusClass}">Jupiter: ${status}</span>`;
    }
}

async function updateAssetUniverse() {
    const data = await fetchJSON('/api/assets');
    if (!data || !data.success) return;
    
    supportedAssets = data.supported_assets || [];
    enabledAssets = data.enabled_assets || [];
    assetsLoaded = true;
    
    const assetsInfo = data.supported_assets_info || [];
    assetsInfo.forEach(info => {
        assetInfo[info.symbol] = info;
    });
    
    const chipsEl = document.getElementById('asset-chips');
    if (!chipsEl) return;
    
    const typeColors = {
        'native': '#14F195',
        'wrapped': '#9945FF',
        'stablecoin': '#3B82F6',
        'lst': '#FF6B35',
        'meme': '#FFD700',
        'governance': '#00BFFF',
        'dex': '#FF69B4'
    };
    
    let html = '';
    for (const info of assetsInfo) {
        const symbol = info.symbol;
        const isEnabled = enabledAssets.includes(symbol);
        const typeColor = typeColors[info.type] || '#666';
        const hlBadge = info.has_hyperliquid ? '<span class="chip-hl">HL</span>' : '';
        const driftBadge = info.has_drift ? '<span class="chip-drift">DR</span>' : '';
        
        html += `
            <label class="asset-chip ${isEnabled ? 'enabled' : ''}" style="--type-color: ${typeColor}">
                <input type="checkbox" value="${symbol}" ${isEnabled ? 'checked' : ''} data-symbol="${symbol}">
                <span class="chip-symbol">${symbol}</span>
                <span class="chip-type">${info.type}</span>
                ${hlBadge}
                ${driftBadge}
            </label>
        `;
    }
    
    chipsEl.innerHTML = html;
    
    // Add event listeners to each checkbox for immediate toggle response
    chipsEl.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', handleAssetToggle);
    });
    
    updateTradeMarketDropdown();
    updatePortfolioBuilderSliders();
}

// Handle individual asset toggle with immediate UI feedback
async function handleAssetToggle(event) {
    const checkbox = event.target;
    const symbol = checkbox.dataset.symbol || checkbox.value;
    const label = checkbox.closest('.asset-chip');
    
    // Update visual state immediately
    if (checkbox.checked) {
        label.classList.add('enabled');
        if (!enabledAssets.includes(symbol)) {
            enabledAssets.push(symbol);
        }
    } else {
        label.classList.remove('enabled');
        enabledAssets = enabledAssets.filter(s => s !== symbol);
    }
    
    // Ensure at least one asset is enabled
    if (enabledAssets.length === 0) {
        checkbox.checked = true;
        label.classList.add('enabled');
        enabledAssets.push(symbol);
        showNotification('At least one asset must be enabled', 'warning');
        return;
    }
    
    // Save to backend
    try {
        const response = await fetch('/api/assets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols: enabledAssets })
        });
        
        if (response.ok) {
            // Refresh UI components that depend on enabled assets
            await updatePrices();
            updatePortfolioBuilderSliders();
            updateTradeMarketDropdown();
            showNotification(`${symbol} ${checkbox.checked ? 'enabled' : 'disabled'}`, 'success');
        } else {
            // Revert on failure
            checkbox.checked = !checkbox.checked;
            if (checkbox.checked) {
                label.classList.add('enabled');
                enabledAssets.push(symbol);
            } else {
                label.classList.remove('enabled');
                enabledAssets = enabledAssets.filter(s => s !== symbol);
            }
            showNotification('Failed to update asset selection', 'error');
        }
    } catch (error) {
        console.error('Error updating asset selection:', error);
        // Revert on error
        checkbox.checked = !checkbox.checked;
        if (checkbox.checked) {
            label.classList.add('enabled');
            enabledAssets.push(symbol);
        } else {
            label.classList.remove('enabled');
            enabledAssets = enabledAssets.filter(s => s !== symbol);
        }
        showNotification('Error updating asset selection', 'error');
    }
}

// Simple notification helper
function showNotification(message, type = 'info') {
    // Check if notification container exists, create if not
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = 'position: fixed; top: 80px; right: 20px; z-index: 1000; display: flex; flex-direction: column; gap: 10px;';
        document.body.appendChild(container);
    }
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        animation: slideIn 0.3s ease;
        ${type === 'success' ? 'background: rgba(20, 241, 149, 0.15); border: 1px solid rgba(20, 241, 149, 0.3); color: #14F195;' : ''}
        ${type === 'error' ? 'background: rgba(255, 64, 129, 0.15); border: 1px solid rgba(255, 64, 129, 0.3); color: #FF4081;' : ''}
        ${type === 'warning' ? 'background: rgba(255, 184, 0, 0.15); border: 1px solid rgba(255, 184, 0, 0.3); color: #FFB800;' : ''}
        ${type === 'info' ? 'background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #3B82F6;' : ''}
    `;
    notification.textContent = message;
    container.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

async function saveAssetSelection() {
    const chipsEl = document.getElementById('asset-chips');
    if (!chipsEl) return;
    
    const selected = [];
    chipsEl.querySelectorAll('input:checked').forEach(input => {
        selected.push(input.value);
    });
    
    if (selected.length === 0) {
        alert('Please select at least one asset');
        return;
    }
    
    const response = await fetch('/api/assets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: selected })
    });
    
    if (response.ok) {
        enabledAssets = selected;
        await updateAssetUniverse();
        await updatePrices();
        updatePortfolioBuilderSliders();
        updateTradeMarketDropdown();
    }
}

function updateTradeMarketDropdown() {
    const tradeAssetSelect = document.getElementById('trade-asset');
    if (!tradeAssetSelect) return;
    
    const currentValue = tradeAssetSelect.value;
    let html = '';
    
    for (const symbol of enabledAssets) {
        if (symbol === 'USDC' || symbol === 'USDT') continue;
        const info = assetInfo[symbol] || {};
        const suffix = info.has_hyperliquid ? ' (HL)' : '';
        html += `<option value="${symbol}">${symbol}${suffix}</option>`;
    }
    
    tradeAssetSelect.innerHTML = html;
    
    if (enabledAssets.includes(currentValue)) {
        tradeAssetSelect.value = currentValue;
    }
    
    const jupiterInputToken = document.getElementById('jupiter-input-token');
    const jupiterOutputToken = document.getElementById('jupiter-output-token');
    
    if (jupiterInputToken && jupiterOutputToken) {
        let jupiterHtml = '';
        for (const symbol of enabledAssets) {
            jupiterHtml += `<option value="${symbol}">${symbol}</option>`;
        }
        jupiterInputToken.innerHTML = jupiterHtml;
        jupiterOutputToken.innerHTML = jupiterHtml;
        
        if (enabledAssets.includes('SOL')) jupiterInputToken.value = 'SOL';
        if (enabledAssets.includes('USDC')) jupiterOutputToken.value = 'USDC';
    }
}

function updatePortfolioBuilderSliders() {
    const buildContainer = document.getElementById('portfolio-sliders');
    if (!buildContainer) return;
    
    // Include ALL enabled assets including stablecoins (USDC/USDT)
    const allAssets = enabledAssets;
    
    if (allAssets.length === 0) {
        buildContainer.innerHTML = '<p class="hint">Select assets to allocate</p>';
        return;
    }
    
    let html = '';
    const defaultWeight = Math.floor(100 / allAssets.length);
    
    for (const symbol of allAssets) {
        const info = assetInfo[symbol] || {};
        const isStablecoin = symbol === 'USDC' || symbol === 'USDT';
        const typeClass = isStablecoin ? 'stablecoin' : '';
        html += `
            <div class="slider-row ${typeClass}">
                <label class="slider-label">
                    <span class="slider-symbol">${symbol}</span>
                    <span class="slider-type">${info.type || ''}</span>
                </label>
                <input type="range" class="portfolio-slider" data-symbol="${symbol}" 
                       min="0" max="100" value="${defaultWeight}">
                <span class="slider-value" id="slider-${symbol}-value">${defaultWeight}%</span>
            </div>
        `;
    }
    
    buildContainer.innerHTML = html;
    
    buildContainer.querySelectorAll('.portfolio-slider').forEach(slider => {
        slider.addEventListener('input', function() {
            updateAssetSliders();
        });
    });
    
    loadPortfolioConfig();
}

async function updateBasisMap() {
    const symbolsParam = enabledAssets.filter(s => s !== 'USDC' && s !== 'USDT').slice(0, 5).join(',');
    const data = await fetchJSON(`/api/venue/basis-map?symbols=${symbolsParam || 'SOL,BTC,ETH'}`);
    if (!data || !data.success) return;
    
    const gridEl = document.getElementById('basis-grid');
    if (!gridEl) return;
    
    let html = '<div class="basis-header"><span></span><span>Spot</span><span>Drift</span><span>HL</span></div>';
    
    for (const [symbol, basisData] of Object.entries(data.basis_map || {})) {
        const driftBps = basisData.drift_basis_bps !== null ? basisData.drift_basis_bps.toFixed(1) : '-';
        const hlBps = basisData.hl_basis_bps !== null ? basisData.hl_basis_bps.toFixed(1) : '-';
        const spotPrice = basisData.spot_price ? formatCurrency(basisData.spot_price) : '-';
        
        html += `
            <div class="basis-row">
                <span class="basis-symbol">${symbol}</span>
                <span class="basis-spot">${spotPrice}</span>
                <span class="basis-drift ${parseFloat(driftBps) > 0 ? 'positive' : parseFloat(driftBps) < 0 ? 'negative' : ''}">${driftBps} bps</span>
                <span class="basis-hl ${parseFloat(hlBps) > 0 ? 'positive' : parseFloat(hlBps) < 0 ? 'negative' : ''}">${hlBps} bps</span>
            </div>
        `;
    }
    
    gridEl.innerHTML = html;
}

async function updateVenueHealth() {
    const data = await fetchJSON('/api/venue/status');
    if (!data || !data.success) return;
    
    const venues = data.venues || {};
    
    const driftHealth = venues.drift?.health_score || 0;
    const hlHealth = venues.hyperliquid?.health_score || 0;
    
    const driftFill = document.getElementById('drift-health-fill');
    const driftPct = document.getElementById('drift-health-pct');
    const hlFill = document.getElementById('hl-health-fill');
    const hlPct = document.getElementById('hl-health-pct');
    
    if (driftFill && driftPct) {
        driftFill.style.width = `${driftHealth}%`;
        driftFill.className = `health-fill ${driftHealth > 70 ? 'good' : driftHealth > 40 ? 'warning' : 'bad'}`;
        driftPct.textContent = `${driftHealth.toFixed(0)}%`;
    }
    
    if (hlFill && hlPct) {
        hlFill.style.width = `${hlHealth}%`;
        hlFill.className = `health-fill ${hlHealth > 70 ? 'good' : hlHealth > 40 ? 'warning' : 'bad'}`;
        hlPct.textContent = `${hlHealth.toFixed(0)}%`;
    }
}

async function updateLatencyBudget() {
    const data = await fetchJSON('/api/hft/latency-budget');
    if (!data || !data.success) return;
    
    const budgetEl = document.getElementById('latency-budget');
    if (!budgetEl) return;
    
    const budget = data.latency_budget || {};
    const target = data.target_total_ms || 500;
    
    let html = '';
    for (const [phase, phaseData] of Object.entries(budget)) {
        const pct = (phaseData.avg_ms / target) * 100;
        html += `
            <div class="budget-row">
                <span class="budget-label">${phase.charAt(0).toUpperCase() + phase.slice(1).replace('_', ' ')}</span>
                <div class="budget-bar-container">
                    <div class="budget-bar" style="width: ${Math.min(pct, 100)}%"></div>
                </div>
                <span class="budget-value">${phaseData.avg_ms.toFixed(0)}ms</span>
            </div>
        `;
    }
    
    budgetEl.innerHTML = html;
    
    const totalEl = document.getElementById('budget-total');
    const targetEl = document.getElementById('budget-target');
    if (totalEl) totalEl.textContent = `${data.total_avg_ms?.toFixed(0) || 0}ms`;
    if (targetEl) targetEl.textContent = `${target}ms`;
}

async function updateChainLogs() {
    const data = await fetchJSON('/api/chain/logs?limit=10');
    if (!data || !data.success) return;
    
    const logsEl = document.getElementById('logs-container');
    if (!logsEl) return;
    
    const logs = data.logs || [];
    
    if (logs.length === 0) {
        logsEl.innerHTML = '<div class="log-empty">No recent logs</div>';
        return;
    }
    
    let html = '';
    for (const log of logs.slice(-10)) {
        const timestamp = new Date((log.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();
        const severity = log.severity || 'info';
        html += `
            <div class="log-entry ${severity}">
                <span class="log-time">${timestamp}</span>
                <span class="log-msg">${log.message || log.signature?.substring(0, 12) + '...' || 'Event'}</span>
            </div>
        `;
    }
    
    logsEl.innerHTML = html;
}

async function updateOracleHealth() {
    const data = await fetchJSON('/api/oracle/pyth-health');
    if (!data || !data.success) return;
    
    const cleanEl = document.getElementById('oracle-clean-count');
    const watchEl = document.getElementById('oracle-watch-count');
    const flaggedEl = document.getElementById('oracle-flagged-count');
    
    if (cleanEl) cleanEl.textContent = data.summary?.clean || 0;
    if (watchEl) watchEl.textContent = data.summary?.watch || 0;
    if (flaggedEl) flaggedEl.textContent = data.summary?.flagged || 0;
    
    const tableEl = document.getElementById('oracle-table');
    if (!tableEl) return;
    
    const symbols = data.symbols || {};
    
    let html = `
        <div class="oracle-row header">
            <span class="col-symbol">Symbol</span>
            <span class="col-pyth">Pyth Price</span>
            <span class="col-composite">Composite</span>
            <span class="col-dev">Deviation</span>
            <span class="col-status">Status</span>
        </div>
    `;
    
    for (const [symbol, info] of Object.entries(symbols)) {
        const pythPrice = info.pyth_price ? `$${info.pyth_price.toFixed(2)}` : '-';
        const composite = info.composite_price ? `$${info.composite_price.toFixed(2)}` : '-';
        const deviation = info.deviation_bps != null ? `${info.deviation_bps.toFixed(1)} bps` : '-';
        const status = info.status || 'unknown';
        
        html += `
            <div class="oracle-row">
                <span class="col-symbol">${symbol}</span>
                <span class="col-pyth">${pythPrice}</span>
                <span class="col-composite">${composite}</span>
                <span class="col-dev">${deviation}</span>
                <span class="col-status"><span class="oracle-status ${status}">${status}</span></span>
            </div>
        `;
    }
    
    tableEl.innerHTML = html;
}

async function updateRouteQuality() {
    const data = await fetchJSON('/api/jupiter/route-quality');
    if (!data || !data.success) return;
    
    const tableEl = document.getElementById('route-quality-table');
    if (!tableEl) return;
    
    const routes = data.route_quality || {};
    
    let html = `
        <div class="route-row header">
            <span class="col-route">Route</span>
            <span class="col-trades">Trades</span>
            <span class="col-fail">Fail Rate</span>
            <span class="col-slip">Avg Slip</span>
            <span class="col-mev">MEV Risk</span>
            <span class="col-score">Score</span>
        </div>
    `;
    
    for (const [routeId, routeData] of Object.entries(routes)) {
        const score = routeData.reliability_score || 0;
        const scoreClass = score > 80 ? 'good' : score > 60 ? 'medium' : 'bad';
        const mevRisk = routeData.mev_risk_level || 'medium';
        
        html += `
            <div class="route-row">
                <span class="col-route">${routeId}</span>
                <span class="col-trades">${routeData.total_trades}</span>
                <span class="col-fail">${(routeData.failure_rate * 100).toFixed(1)}%</span>
                <span class="col-slip">${routeData.avg_slippage_bps?.toFixed(1) || 0} bps</span>
                <span class="col-mev"><span class="mev-badge ${mevRisk}">${mevRisk}</span></span>
                <span class="col-score ${scoreClass}">${score.toFixed(0)}</span>
            </div>
        `;
    }
    
    tableEl.innerHTML = html;
}

function initNewPanels() {
    const fetchRoutesBtn = document.getElementById('fetch-jupiter-routes');
    if (fetchRoutesBtn) {
        fetchRoutesBtn.addEventListener('click', fetchJupiterRoutes);
    }
    
    const execModeSelect = document.getElementById('exec-mode-select');
    if (execModeSelect) {
        execModeSelect.addEventListener('change', async (e) => {
            const mode = e.target.value;
            await fetch('/api/execution/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
        });
    }
    
    const saveAssetsBtn = document.getElementById('save-assets');
    if (saveAssetsBtn) {
        saveAssetsBtn.addEventListener('click', saveAssetSelection);
    }
}

async function init() {
    console.log('Initializing Solana Yield Orchestrator Dashboard...');
    
    initSSE();
    initChartControls();
    initTradeTicket();
    initSimSettings();
    initPositionsTabs();
    initPortfolioBuilder();
    initNewPanels();
    
    // CRITICAL: Load asset universe FIRST to populate enabledAssets before other updates
    await updateAssetUniverse();
    
    // Now run all other updates in parallel
    await Promise.all([
        updateStatus(),
        updatePrices(),
        updateMetrics(),
        updatePositions(),
        updatePerpRisk(),
        updatePortfolioState(),
        updateChainState(),
        updatePriorityFees(),
        updateFundingTermStructure(),
        updateMarginHeatmap(),
        updateHFTLatency(),
        updateCrossVenueFunding(),
        updateBasisMap(),
        updateVenueHealth(),
        updateLatencyBudget(),
        updateChainLogs(),
        updateRouteQuality(),
        updateOracleHealth()
    ]);
    
    document.getElementById('run-simulation').addEventListener('click', runSimulation);
    
    setInterval(updateStatus, UPDATE_INTERVAL);
    setInterval(updatePrices, UPDATE_INTERVAL);
    setInterval(updateMetrics, UPDATE_INTERVAL);
    setInterval(updatePositions, UPDATE_INTERVAL * 2);
    setInterval(updatePerpRisk, UPDATE_INTERVAL * 2);
    setInterval(updatePortfolioState, UPDATE_INTERVAL * 2);
    setInterval(updateChainState, UPDATE_INTERVAL);
    setInterval(updatePriorityFees, UPDATE_INTERVAL * 3);
    setInterval(updateFundingTermStructure, UPDATE_INTERVAL * 2);
    setInterval(updateMarginHeatmap, UPDATE_INTERVAL * 2);
    setInterval(updateHFTLatency, UPDATE_INTERVAL * 2);
    setInterval(updateCrossVenueFunding, UPDATE_INTERVAL * 3);
    setInterval(updateAssetUniverse, UPDATE_INTERVAL * 4);
    setInterval(updateBasisMap, UPDATE_INTERVAL * 2);
    setInterval(updateVenueHealth, UPDATE_INTERVAL * 3);
    setInterval(updateLatencyBudget, UPDATE_INTERVAL * 3);
    setInterval(updateChainLogs, UPDATE_INTERVAL);
    setInterval(updateRouteQuality, UPDATE_INTERVAL * 4);
    setInterval(updateOracleHealth, UPDATE_INTERVAL * 3);
    
    console.log('Dashboard initialized successfully');
}

document.addEventListener('DOMContentLoaded', init);
