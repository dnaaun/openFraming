duration = 250
n = 12
k = 10

// Data processing steps
function get_datevalue(data){
    datevalues = Array.from(d3.rollup(data, ([d]) => d.value, d => d.date, d=>d.name))
  .sort(([a], [b]) => d3.ascending(a, b))
    return datevalues;
}

function rank(value) {
    const data = Array.from(names, name => ({name, value: value(name)}));
    data.sort((a, b) => d3.descending(a.value, b.value));
    for (let i = 0; i < data.length; ++i) data[i].rank = Math.min(n, i);
    return data;
}
 
function parse2(dae){
    var parse = d3.timeParse("%Y-%m-%d");
    return parse(dae);
}

function get_keyframes(datevalues, names){
    const keyframes = [];
    let ka, a, kb, b;
    for ([[ka, a], [kb, b]] of d3.pairs(datevalues)) {
        for (let i = 0; i < k; ++i) {
        const t = i / k;
        keyframes.push([
            new Date(ka * (1 - t) + kb * t),
            rank(name => (a.get(name) || 0) * (1 - t) + (b.get(name) || 0) * t)
        ]);
        }
    }
    keyframes.push([new Date(kb), rank(name => b.get(name) || 0)]);
    return keyframes;
}

// Plotting functions
function bars(svg, prev, next) {
    let bar = svg.append("g")
      .attr("fill-opacity", 0.7)
      .selectAll("rect");
  
    return ([date, data], transition) => bar = bar
      .data(data.slice(0, n), d => d.name)
      .join(
        enter => enter.append("rect")
          .attr("fill", color(data))
          .attr("height", y.bandwidth())
          .attr("x", x(0))
          .attr("y", d => y((prev.get(d) || d).rank))
          .attr("width", d => x((prev.get(d) || d).value) - x(0)),
        update => update,
        exit => exit.transition(transition).remove()
          .attr("y", d => y((next.get(d) || d).rank))
          .attr("width", d => x((next.get(d) || d).value) - x(0))
      )
      .call(bar => bar.transition(transition)
        .attr("y", d => y(d.rank))
        .attr("width", d => x(d.value) - x(0)));
}

function labels(svg, x, y) {
  let label = svg.append("g")
      .style("font", "bold 15px var(--sans-serif)")
      .style("font-variant-numeric", "tabular-nums")
      .attr("text-anchor", "end")
    .selectAll("text");

  return ([date, data], transition) => label = label
    .data(data.slice(0, n), d => d.name)
    .join(
      enter => enter.append("text")
        .attr("transform", d => `translate(${x((prev.get(d) || d).value)},${y((prev.get(d) || d).rank)})`)
        .attr("y", y.bandwidth() / 2)
        .attr("x", -6)
        .attr("dy", "-0.25em")
        .text(d => d.name)
        .attr("font-weight", "bold")
        .call(text => text.append("tspan")
          .attr("fill-opacity", 0.7)
          .attr("font-weight", "normal")
          .attr("x", -6)
          .attr("dy", "1.15em")),
      update => update,
      exit => exit.transition(transition).remove()
        .attr("transform", d => `translate(${x((next.get(d) || d).value)},${y((next.get(d) || d).rank)})`)
        .call(g => g.select("tspan").tween("text", d => textTween(d.value, (next.get(d) || d).value)))
    )
    .call(bar => bar.transition(transition)
      .attr("transform", d => `translate(${x(d.value)},${y(d.rank)})`)
      .call(g => g.select("tspan").tween("text", d => textTween((prev.get(d) || d).value, d.value))))
}

formatNumber = d3.format(",d")
function textTween(a, b) {
    const i = d3.interpolateNumber(a, b);
    return function(t) {
      this.textContent = formatNumber(i(t));
    };
}

function axis(svg, x, y, margin) {
    const g = svg.append("g")
        .attr("transform", "translate(0,"+margin.top.toString()+")");
  
    const axis = d3.axisTop(x)
        .ticks(width / 160)
        .tickSizeOuter(0)
        .tickSizeInner(-barSize * (n + y.padding()));
  
    return (_, transition) => {
      g.transition(transition).call(axis);
      g.select(".tick:first-of-type text").remove();
      g.selectAll(".tick:not(:first-of-type) line").attr("stroke", "white");
      g.select(".domain").remove();
    };
}

formatDate = d3.utcFormat("%m - %d - %Y")
function ticker(svg, barSize, keyframes) {
    const titl = svg.append("text")
        .style("font-size","14px")
        .style("font-variant-numeric", "tabular-nums")
        .attr("text-anchor", "end")
        .attr("x", width - 100)
        .attr("y", margin.top + barSize * (n - 0.45) - 30)
        .attr("dy", "0.32em")
        .text("Date :");
    const now = svg.append("text")
        .style("font-size", (barSize - 10).toString()+"px")
        .style("font-weight", "bold")
        .style("font-variant-numeric", "tabular-nums")
        .attr("text-anchor", "end")
        .attr("x", width - 5)
        .attr("y", margin.top + barSize * (n - 0.45))
        .attr("dy", "0.32em")
        .text(formatDate(keyframes[0][0]));
    return ([date], transition) => {
      transition.end().then(() => now.text(formatDate(date)));
    };
  }

function color(data){
  const scale = d3.scaleOrdinal(d3.schemePaired);
  if (data.some(d => d.category !== undefined)) {
    const categoryByName = new Map(data.map(d => [d.name, d.category])) 
    scale.domain(Array.from(categoryByName.values()));
    return d => scale(categoryByName.get(d.name));
  };
  return d => scale(d.name);
}




// viewof replay = html`<button>Replay`;
async function chart(data, keyframes, prev, next){
    // replay;
    margin = ({top: 16, right: 6, bottom: 6, left: 0});
    barSize = 40;
    height = margin.top + barSize * n + margin.bottom;
    width = 800
    
    x = d3.scaleLinear([0, 1], [margin.left, width - margin.right]);
    y = d3.scaleBand()
    .domain(d3.range(n + 1))
    .rangeRound([margin.top, margin.top + barSize * (n + 1 + 0.1)])
    .padding(0.1);

    const svg = d3.select("body").append("svg")
      .attr("width", width)
      .attr("height", height);
    
  
    const updateBars = bars(svg, prev, next);
    const updateAxis = axis(svg, x, y, margin);
    const updateLabels = labels(svg, x, y);
    const updateTicker = ticker(svg, barSize, keyframes);
    
    
    
    // yield svg.node();
  
    for (const keyframe of keyframes) {
      const transition = svg.transition()
          .duration(duration)
          .ease(d3.easeLinear);
      // Extract the top bar’s value.
      x.domain([0, keyframe[1][0].value]);
  
      updateAxis(keyframe, transition);
      updateBars(keyframe, transition);
      updateLabels(keyframe, transition);
      updateTicker(keyframe, transition);
  
      // invalidation.then(() => svg.interrupt());
      await transition.end();
    }
  }

d3.csv("category-brands.csv", function(d) {
    
    return {
      date : parse2(d.date),
      name : d.name,
      category : d.category,
      value : +d.value
    };
  }).then(function(data){
    names = new Set(data.map(d => d.name));
    dat = get_datevalue(data);
    keyframes = get_keyframes(dat,  names);
    nameframes = d3.groups(keyframes.flatMap(([, data]) => data), d => d.name);
    prev = new Map(nameframes.flatMap(([, data]) => d3.pairs(data, (a, b) => [b, a])));
    next = new Map(nameframes.flatMap(([, data]) => d3.pairs(data)));
    chart(data, keyframes, prev, next);
});

