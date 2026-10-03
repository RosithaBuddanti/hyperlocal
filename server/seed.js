const bcrypt = require('bcryptjs');
const { initDB, dbRun, dbGet, dbAll } = require('./db');

async function seedDatabase() {
  await initDB();

  // Check if already seeded
  const existingUser = await dbGet('SELECT id FROM users LIMIT 1');
  if (existingUser) {
    console.log('Database already has records. Seeding skipped.');
    return;
  }

  console.log('Seeding demo accounts (3 citizens, 10 responders, 3 dispatchers/admins)...');
  const passwordHash = await bcrypt.hash('password123', 10);

  // 1. Citizens (3 Users)
  const citizens = [
    { email: 'aarav@demo.com', full_name: 'Aarav Sharma', phone: '+91 98765 43210', emergency_contact_phone: '+91 98765 00001' },
    { email: 'neha@demo.com', full_name: 'Neha Patel', phone: '+91 98765 43214', emergency_contact_phone: '+91 98765 00002' },
    { email: 'rohan@demo.com', full_name: 'Rohan Verma', phone: '+91 98765 43215', emergency_contact_phone: '+91 98765 00003' }
  ];

  for (const c of citizens) {
    await dbRun(
      `INSERT INTO users (email, password_hash, full_name, phone, role, emergency_contact_phone)
       VALUES (?, ?, ?, ?, 'citizen', ?)`,
      [c.email, passwordHash, c.full_name, c.phone, c.emergency_contact_phone]
    );
  }

  // 2. Responders (10 Users + 10 Responder Profiles)
  const responders = [
    { email: 'ambulance1@demo.com', name: 'Capt. Rajesh Kumar (EMS Alpha 108)', phone: '+91 98765 43221', service: 'Ambulance', lat: 12.9760, lng: 77.6010, vehicle: 'KA-01-AMB-108', org: 'City Memorial EMS Fleet' },
    { email: 'ambulance2@demo.com', name: 'Paramedic Sunita Rao (EMS Bravo 104)', phone: '+91 98765 43222', service: 'Ambulance', lat: 12.9810, lng: 77.5920, vehicle: 'KA-01-AMB-104', org: 'Victoria Hospital Trauma Service' },
    { email: 'ambulance3@demo.com', name: 'Dr. Vikram Seth (Trauma Mobile Unit)', phone: '+91 98765 43223', service: 'Ambulance', lat: 12.9640, lng: 77.6050, vehicle: 'KA-01-AMB-999', org: 'Apex Cardiac Response' },
    
    { email: 'police1@demo.com', name: 'Inspector Priya Singh (Patrol 01)', phone: '+91 98765 43224', service: 'Police', lat: 12.9785, lng: 77.5910, vehicle: 'KA-01-POL-01', org: 'Central City Police Station' },
    { email: 'police2@demo.com', name: 'Officer Amit Deshmukh (Highway Patrol 12)', phone: '+91 98765 43225', service: 'Police', lat: 12.9680, lng: 77.5840, vehicle: 'KA-01-POL-12', org: 'Traffic & Highway Patrol Division' },
    { email: 'police3@demo.com', name: 'Sub-Inspector Kavita Joshi (PCR Van 07)', phone: '+91 98765 43226', service: 'Police', lat: 12.9710, lng: 77.6150, vehicle: 'KA-01-POL-07', org: 'East District Patrol Unit' },

    { email: 'fire1@demo.com', name: 'Station Officer Suresh Nair (Tender 09)', phone: '+91 98765 43227', service: 'Fire', lat: 12.9640, lng: 77.5850, vehicle: 'KA-01-FIRE-09', org: 'MG Road Fire & Rescue HQ' },
    { email: 'fire2@demo.com', name: 'Firefighter Deepak Pillai (Quick Fire 04)', phone: '+91 98765 43228', service: 'Fire', lat: 12.9840, lng: 77.6080, vehicle: 'KA-01-FIRE-04', org: 'North Zone Fire Command' },

    { email: 'rescue1@demo.com', name: 'Rescue Lead Manoj Gowda (NDRF Disaster)', phone: '+91 98765 43229', service: 'Rescue', lat: 12.9550, lng: 77.5700, vehicle: 'KA-01-RSC-88', org: 'State Disaster Response Force' },
    { email: 'rescue2@demo.com', name: 'Specialist Ananya Sen (Urban Search)', phone: '+91 98765 43230', service: 'Rescue', lat: 12.9880, lng: 77.6200, vehicle: 'KA-01-RSC-42', org: 'Specialized Urban Rescue Squad' }
  ];

  for (const r of responders) {
    const res = await dbRun(
      `INSERT INTO users (email, password_hash, full_name, phone, role)
       VALUES (?, ?, ?, ?, 'responder')`,
      [r.email, passwordHash, r.name, r.phone]
    );

    await dbRun(
      `INSERT INTO responders (user_id, service_type, is_available, is_verified, lat, lng, vehicle_number, organization_name)
       VALUES (?, ?, 1, 1, ?, ?, ?, ?)`,
      [res.lastID, r.service, r.lat, r.lng, r.vehicle, r.org]
    );
  }

  // 3. Dispatchers / Admins (3 Users)
  const dispatchers = [
    { email: 'admin@demo.com', name: 'Chief Dispatcher Rajesh Mehra', phone: '+91 98765 43291' },
    { email: 'dispatcher2@demo.com', name: 'Supervisor Rahul Roy (Sector North)', phone: '+91 98765 43292' },
    { email: 'dispatcher3@demo.com', name: 'Commander Meera Reddy (Crisis Operations)', phone: '+91 98765 43293' }
  ];

  for (const d of dispatchers) {
    await dbRun(
      `INSERT INTO users (email, password_hash, full_name, phone, role)
       VALUES (?, ?, ?, ?, 'admin')`,
      [d.email, passwordHash, d.name, d.phone]
    );
  }

  // 4. Area Broadcast Alert
  await dbRun(
    `INSERT INTO area_alerts (title, message, alert_type, radius_km, center_lat, center_lng)
     VALUES (?, ?, ?, ?, ?, ?)`,
    [
      '⚠️ High Alert: Flash Flood Warning in Low-Lying Sector 4',
      'Heavy water accumulation reported along Ring Road. Avoid subways and ground-level underpasses. Rescue teams mobilized.',
      'Warning',
      5.0,
      12.9600,
      77.5900
    ]
  );

  // 6. Emergency Contacts
  const contacts = [
    { name: '112 Unified National Helpline', phone: '112', service_type: 'General', address: 'All India Coordination Center', lat: 12.9716, lng: 77.5946 },
    { name: '108 Central Emergency Medical Services', phone: '108', service_type: 'Ambulance', address: 'Central Hospital Depot', lat: 12.9650, lng: 77.5750 },
    { name: 'Victoria Govt Hospital Trauma Center', phone: '+91 80 2670 1150', service_type: 'Ambulance', address: 'Fort Road, near City Market', lat: 12.9628, lng: 77.5752 },
    { name: '100 City Police Control Room', phone: '100', service_type: 'Police', address: 'Police HQ, Infantry Road', lat: 12.9810, lng: 77.5990 },
    { name: '101 Fire & Emergency Command', phone: '101', service_type: 'Fire', address: 'State Fire Services HQ', lat: 12.9660, lng: 77.5870 },
    { name: '1077 State Disaster Response Helpline', phone: '1077', service_type: 'Rescue', address: 'Disaster Management Cell', lat: 12.9780, lng: 77.5910 }
  ];

  for (const c of contacts) {
    await dbRun(
      `INSERT INTO contacts (name, phone, service_type, address, lat, lng)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [c.name, c.phone, c.service_type, c.address, c.lat, c.lng]
    );
  }

  console.log('Demo database seeded successfully with all required roles and responders!');
}

if (require.main === module) {
  seedDatabase().catch(console.error);
}

module.exports = seedDatabase;
