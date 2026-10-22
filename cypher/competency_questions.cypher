// WaterOnt competency questions in Cypher, for the graph pushed by
//   python -m wateront build --inp L-TOWN.inp --neo4j bolt://localhost:7687
// Node ids are type-prefixed: J:<junction>, P:<pipe>, PU:<pump>, V:<valve>.

// CQ1  What nodes does pipe $P connect?
MATCH (:Pipe {name: $P})-[:STARTS_AT|ENDS_AT]->(n) RETURN n.id;

// CQ2  Which pipes are incident on junction $J and what are their diameters?
MATCH (p:Pipe)-[:STARTS_AT|ENDS_AT]->(:Junction {name: $J}) RETURN p.name, p.diameter;

// CQ4  Which valves are PRVs?
MATCH (v:PressureReducingValve) RETURN v.name, v.valveSetting;

// CQ7  Which links are currently closed and why?
MATCH (l:Link)-[:HAS_STATUS]->(s:OperationalStatus {value: 'Closed'}) RETURN l.name, s.reason, s.controlledBy;

// Asset: cast iron pipes installed before 1970 with more than two failures
MATCH (p:Pipe)-[:HAS_ASSET_RECORD]->(a:AssetRecord {material: 'CI'})
WHERE a.installationYear < 1970 AND a.failureCount > 2
RETURN p.name, a.installationYear, a.failureCount;

// Risk: two-hop band pattern, no numeric thresholds
// hydraulically critical AND high/very-high failure probability
MATCH (p:Pipe)-[:HAS_RISK_PROFILE]->(rp)-[:HAS_HYDRAULIC_CRITICALITY]->(:HydraulicCriticality {band: 'Critical'}),
      (rp)-[:HAS_FAILURE_PROBABILITY]->(fp:FailureProbability)
WHERE fp.band IN ['High', 'VeryHigh']
RETURN p.name, rp.failureProbability ORDER BY rp.priority DESC;

// Risk: backbone pipes serving a critical service zone with intolerable risk
MATCH (p:Pipe)-[:HAS_RISK_PROFILE]->(rp)
MATCH (rp)-[:HAS_SERVICE_CRITICALITY]->(:ServiceCriticality {band: 'Critical'})
MATCH (rp)-[:HAS_OVERALL_RISK]->(:OverallRisk {band: 'Intolerable'})
RETURN p.name;

// Decision: which pipes are recommended for replacement, and which clause authorises it?
MATCH (d:DecisionRecord)-[:RECOMMENDS]->(:MaintenanceAction {actionType: 'Replacement'}),
      (d)-[:CONCERNS]->(p:Pipe), (d)-[:SUPPORTED_BY]->(g:EngineeringGuideline)
RETURN p.name, collect(g.id + ' (' + g.framework + ')') AS authorisedBy ORDER BY d.priority DESC;

// Full audit trail for one recommendation
MATCH path = (d:DecisionRecord)-[:CONCERNS]->(:Pipe {name: $P})
OPTIONAL MATCH ev = (d)-[:BASED_ON|SUPPORTED_BY|RECOMMENDS]->()
RETURN path, collect(ev);
