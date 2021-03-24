/* * * * */
/*  DATA */
/* * * * */
let GUN_DATA;
let COV_US;
let COV_KO;

let pie;
// let datevalues;

let gun_names;
let cov_us_names;
let cov_ko_names;


//  Racing Bar chart
let duration = 250
let n = 12
let k = 15

// Data processing steps
function get_datevalue(data){
    let datevalues = Array.from(d3.rollup(data, ([d]) => d.value, d => d.date, d=>d.name))
        .sort(([a], [b]) => d3.ascending(a, b))
    return datevalues;
}

function rank(names, value) {
    const data = Array.from(names, name => ({name, value: value(name)}));
    data.sort((a, b) => d3.descending(a.value, b.value));
    for (let i = 0; i < data.length; ++i) data[i].rank = Math.min(n, i);
    return data;
}

function parse2(dae){
    var parse = d3.timeParse("%Y-%m-%d");
    return parse(dae);
}
function parse3(dae){
    var parse = d3.timeParse("%m-%d-%Y");
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
                rank(names,name => (a.get(name) || 0) * (1 - t) + (b.get(name) || 0) * t)
            ]);
        }
    }
    console.log(b);
    keyframes.push([new Date(kb), rank(names,(name) => {
        return b.get(name) || 0;
    })]);
    return keyframes;
}

// Plotting functions
function bars(svg, prev, next, data2, x, y) {
    let bar = svg.append("g")
        .attr("fill-opacity", 0.8)
        .selectAll("rect");

    return ([date, data], transition) => bar = bar
        .data(data.slice(0, n), d => d.name)
        .join(
            enter => enter.append("rect")
                .attr("fill", color(data2))
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

function labels(svg, x, y, prev, next) {
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
                .attr("y", y.bandwidth() /2)
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

let formatNumber = d3.format(",d")
function textTween(a, b) {
    const i = d3.interpolateNumber(a, b);
    return function(t) {
        this.textContent = i(t).toFixed(2) + '%';
    };
}

function axis(svg, x, y, margin, width, barSize) {
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

let formatDate = d3.utcFormat("%m - %d - %Y")
function ticker(svg, barSize, keyframes, width) {
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
    return data => scale(data.name);
}




// viewof replay = html`<button>Replay`;
async function chart(data, keyframes, prev, next, arg){
    // replay;
    margin = ({top: 16, right: 6, bottom: 6, left: 0});
    let barSize = 40;
    let height = margin.top + barSize * n + margin.bottom;
    let width = 800

    let x = d3.scaleLinear([0, 1], [margin.left, width - margin.right]);
    let y = d3.scaleBand()
        .domain(d3.range(n + 1))
        .rangeRound([margin.top, margin.top + barSize * (n + 1 + 0.1)])
        .padding(0.1);

    const svg = d3.select(`#cov-${arg}-chart`).append("svg")
        .attr("width", width)
        .attr("height", height);


    const updateBars = bars(svg, prev, next, data, x, y);
    const updateAxis = axis(svg, x, y, margin, width, barSize);
    const updateLabels = labels(svg, x, y, prev, next);
    const updateTicker = ticker(svg, barSize, keyframes, width);

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

function plot_chart(arg){
    let data;
    if (arg==='us') {
        data = COV_US;
    } else {
        data = COV_KO;
    }
    let names = new Set(data.map(d => d.name));
    let dat = get_datevalue(data);
    console.log(dat);
    let keyframes = get_keyframes(dat,  names);
    let nameframes = d3.groups(keyframes.flatMap(([, data]) => data), d => d.name);
    let prev = new Map(nameframes.flatMap(([, data]) => d3.pairs(data, (a, b) => [b, a])));
    let next = new Map(nameframes.flatMap(([, data]) => d3.pairs(data)));
    chart(data, keyframes, prev, next, arg);
}

function remove_active_from_tabs(){
    $('.covid_tab_us a').removeClass('active');
    $('.covid_tab_ko a').removeClass('active');
    $('.gun_violence a').removeClass('active');
}
function show_covid(arg){
    // remove_active_from_tabs()
    // $('.covid_tab_'+arg+' a').addClass('active');
    // $('.gunviolence_div').hide();
    // $(`#cov-${arg}-chart`)[0].innerHTML = arg+ '<br>'
    plot_chart(arg)
}

// Gunviolence Static Bar chart

var pie_width = null;
let pie_height = null;
let pie_margin = null;

var bar_width = null
let bar_height = null;
var margin = null;

var radius = null;
var svg = null;
var svg_bar = null;
var color_pie = null;
var data_processing = null

function process_data(data, data_processing){
    let count_category = []
    if(data_processing.indexOf('left')!=-1){
        temp = data.filter(x=>{
            if(x.leaning=='Left'){
                return x
            }
        });
        count_category = count_category.concat(temp);
    }
    if(data_processing.indexOf('right')!=-1){
        temp = data.filter(x=>{
            if(x.leaning=='Right'){
                return x
            }
        });
        count_category = count_category.concat(temp);
    }
    if(data_processing.indexOf('neutral')!=-1){
        temp = data.filter(x=>{
            if(x.leaning=='Neutral'){
                return x
            }
        });
        count_category = count_category.concat(temp);
    }

    if(data_processing.indexOf('2016')!=-1){
        temp = data.filter(x=>{
            if(x.date.getFullYear()==2016){
                return x
            }
        });
        count_category = count_category.concat(temp);
    }

    if(data_processing.indexOf('2017')!=-1){
        temp = data.filter(x=>{
            if(x.date.getFullYear()==2017){
                return x
            }
        });
        count_category = count_category.concat(temp);
    }

    if(data_processing.indexOf('2018')!=-1){
        temp = data.filter(x=>{
            if(x.date.getFullYear()==2018){
                return x
            }
        });
        count_category = count_category.concat(temp);
    }

    if(count_category.length==0){
        count_category = data
    }
    let categories_count = count_category.map(x=>x.category).reduce(function(countMap, word) {countMap[word] = ++countMap[word] || 1;return countMap}, {})
    let categories_month = count_category
        .map(x=>[x.date.getMonth(), x.category])
        .reduce((r, [v, k]) => {
            if(!r[v]) r[v] = {};
            r[v][k] = ++r[v][k] || 1;
            return r;
        }, {})
    return [categories_count, categories_month];
}


function update(data, color) {
    // Compute the position of each group on the pie:
    var temp = new Map(Object.entries(data))
    let total_temp = Object.entries(data).map(x => x[1]).reduce((x, y)=>x + y, 0)
    var data_ready = pie(Array.from(temp, ([nam, value])=>({'key': nam, 'value': value})))
    data_ready.sort(function(a, b) {
        var nameA = a.data.key.toUpperCase(); // ignore upper and lowercase
        var nameB = b.data.key.toUpperCase(); // ignore upper and lowercase
        if (nameA < nameB) {
            return -1;
        }
        if (nameA > nameB) {
            return 1;
        }
        // names must be equal
        return 0;
    })
    // console.log('heree', data_ready);
    // map to data
    // var u = svg.selectAll("path")
    //   .data(data_ready)
    var arc = d3.arc()
        .innerRadius(radius * 0.5)         // This is the size of the donut hole
        .outerRadius(radius * 0.8)

    var outerArc = d3.arc()
        .innerRadius(radius * 0.9)
        .outerRadius(radius * 0.9)
    // Build the pie chart: Basically, each part of the pie is a path that we build using the arc function.
    svg.selectAll("*").remove();
    svg
        .selectAll('allSlices')
        .data(data_ready)
        .enter()
        .insert('path')
        .attr('d', arc)
        .attr('fill', function(d){ return(color(d.data.key)) })
        .attr("stroke", "white")
        .style("stroke-width", "2px")
        .style("opacity", 0.7)
        .transition().duration(1000)
        .attrTween("d", function(d) {
            this._current = this._current || d;
            var interpolate = d3.interpolate(this._current, d);
            this._current = interpolate(0);
            return function(t) {
                return arc(interpolate(t));
            };
        })

    svg
        .selectAll('allPolylines')
        .data(data_ready)
        .enter()
        .insert('polyline')
        .attr("stroke", "black")
        .style("fill", "none")
        .attr("stroke-width", 1)
        .transition().duration(1000)
        .attrTween("points", function(d){
            this._current = this._current || d;
            var interpolate = d3.interpolate(this._current, d);
            this._current = interpolate(0);
            return function(t) {
                var d2 = interpolate(t);
                var pos = outerArc.centroid(d2);
                pos[0] = radius * 0.95 * (midAngle(d2) < Math.PI ? 1 : -1);
                return [arc.centroid(d2), outerArc.centroid(d2), pos];
            };
        });


    function midAngle(d){
        return d.startAngle + (d.endAngle - d.startAngle)/2;
    }

    svg
        .selectAll('allLabels')
        .data(data_ready)
        .enter()
        .insert('text')
        .text( function(d) { return (d.data.key + ' (' + (d.data.value/total_temp*100).toFixed(2)+ '%)') } )
        .style("font-size", 10.5)
        .transition().duration(1000)
        .attrTween("transform", function(d) {
            this._current = this._current || d;
            var interpolate = d3.interpolate(this._current, d);
            this._current = interpolate(0);
            return function(t) {
                var d2 = interpolate(t);
                var pos = outerArc.centroid(d2);
                pos[0] = radius * (midAngle(d2) < Math.PI ? 1 : -1);
                return "translate("+ pos +")";
            };
        })
        .styleTween("text-anchor", function(d){
            this._current = this._current || d;
            var interpolate = d3.interpolate(this._current, d);
            this._current = interpolate(0);
            return function(t) {
                var d2 = interpolate(t);
                return midAngle(d2) < Math.PI ? "start":"end";
            };
        })
}

function create_bar(data2, color_bar){
    $('#gunv-bars').html('By Month <br>');
    svg_bar = d3.select("#gunv-bars")
        .append("svg")
        .attr("width", bar_width + margin.left + margin.right)
        .attr("height", bar_height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let data = []
    let max_val = -1
    let color_codes = []
    let flag = 0;
    var months = Object.keys(data2);
    var entries = Object.entries(data2);
    for(let i = 0;i<months.length;i++){
        let temp = {}
        temp['month'] = i;
        let temp_val = 0;
        for(let entry of Object.keys(data2[0]).sort()){
            temp[entry] = entries[i][1][entry];
            temp_val += entries[i][1][entry];
            if(flag==0) color_codes.push(color_bar(entry));
        }
        flag = 1;
        if(temp_val>max_val) max_val = temp_val;
        data.push(temp);
    }
    var parse = d3.format("m");
    var subgroups = Object.keys(data[0]).slice(1)
    // console.log(subgroups)
    var groups = data.map(x => {return x.month})
    var x = d3.scaleBand()
        .domain(groups)
        .range([0, bar_width])
        .padding([0.2])

    svg_bar.append("g")
        .attr("transform", "translate(0," + bar_height + ")")
        .call(d3.axisBottom(x).tickSizeOuter(0));

    var y = d3.scaleLinear()
        .domain([0, max_val])
        .range([ bar_height, 0 ]);

    svg_bar.append("g")
        .call(d3.axisLeft(y));


    var stackedData = d3.stack()
        .keys(subgroups)
        (data);
    // console.log(stackedData);

    svg_bar.append("g")
        .selectAll("g")
        // Enter in the stack data = loop key per key = group per group
        .data(stackedData)
        .enter().append("g")
        .attr("fill", function(d) {return color_bar(d.key); })
        .attr("opacity", 0.7)
        .selectAll("rect")
        // enter a second time = loop subgroup per subgroup to add all rectangles
        .data(function(d) { return d; })
        .enter().append("rect")
        .attr("x", function(d) {return x(d.data.month); })
        .attr("y", function(d) { return y(d[1]); })
        .attr("height", function(d) { return y(d[0]) - y(d[1]); })
        .attr("width",x.bandwidth());

    var legend = svg_bar.selectAll(".legend")
        .data(color_codes)
        .enter().append("g")
        .attr("class", "legend")
        .attr("transform", function(d, i) { return "translate(30," + i * 19 + ")"; });

    legend.append("text")
        .attr("x", bar_width -120)
        .attr("y", 9)
        .attr("dy", ".10em")
        .style("text-anchor", "start")
        .style("font-size", "14")
        .text(function(d, i) {
            return Object.keys(data2[0]).sort()[i];
        });
    legend.append("rect")
        .attr("x", bar_width - 140)
        .attr("width", 12)
        .attr("height", 12)
        .style("fill", function(d, i) {return color_codes.slice().reverse()[i];})
        .style("opacity", 0.7);


}

function add_names_to_div(names){
    // console.log(names);
    for(let name of names){
        $('#news-list').append(
            "<li class='list-group-item'>"+name+"</li>"
        )
    }
}

function plot_pie_chart(data_processing){
    var some = process_data(GUN_DATA, data_processing);
    if(color_pie==null){
        color_pie = d3.scaleOrdinal()
            .domain(Object.keys(some))
            .range(d3.schemeCategory10);
    }
    update(some[0], color_pie)
    create_bar(some[1], color_pie);
}

function show_gunviolence(val='all'){
    // document.getElementsByClassName('chart')[0].innerHTML ='Gunviolence <br>'
    data_processing = []
    $('.bar-graph').html('By Month <br>')

    pie_width = 800,
        pie_height = 800,
        pie_margin = 170;

    margin = {top: 10, right: 80, bottom: 20, left: 50},
        bar_width = 550,
        bar_height = 500;

    radius = Math.min(pie_width, pie_height) / 2 - pie_margin
    pie = d3.pie()
        .value(function(d) {return d.value; })
        .sort(function(a, b) {return d3.ascending(a.key, b.key);} ) // This make sure that group order remains the same in the pie chart
    svg = d3.select("#gunv-chart")
        .append("svg")
        .attr("width", pie_width)
        .attr("height", pie_height)
        .append("g")
        .attr("transform", "translate(" + pie_width / 2 + "," + pie_height / 2 + ")");

    plot_pie_chart(data_processing)
}

function check_clicked(val){
    if($('#'+val).prop('checked')===true){
        data_processing.push(val)
    }else{
        index = data_processing.indexOf(val)
        data_processing.splice(index)
    }
    plot_pie_chart(data_processing)
}


/* * * * * * */
/*  BROWSER  */
/* * * * * * */
$(document).ready(function() {

    // $("#step4")

    d3.csv("../frontend/datasets/gunviolence_data.csv",function(d) {
        return {
            date : parse3(d.date),
            name : d.name,
            title : d.title,
            leaning : d.leaning,
            category : d.category,
        };
    }).then(function(data) {
        gun_names = new Set(data.map(d => d.name));
        add_names_to_div(gun_names);
        GUN_DATA=data;
        show_gunviolence();
    });

    d3.csv("../frontend/datasets/covid19_US.csv", function(d) {
        return {
            date : parse2(d.date),
            name : d.name,
            category : d.category,
            value : +d.value
        };
    }).then(function(data) {
        COV_US=data;
        show_covid('us');
    });

    d3.csv("../frontend/datasets/covid19_KO.csv",function(d) {
        return {
            date : parse2(d.date),
            name : d.name,
            category : d.category,
            value : +d.value
        };
    }).then(function(data) {
        COV_KO=data;
        show_covid('ko');
    });

});
