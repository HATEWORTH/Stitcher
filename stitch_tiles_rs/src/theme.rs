//! Stitcher UI Theme — Trilithium / brushed-metal style
//!
//! All colors, bevel drawing, and widget helpers live here.
//! To reskin the app, edit the palette constants and widget functions below.
//! The rest of the app (`main.rs`) only references items from this module.

use eframe::egui;

// ─────────────────────────────────────────────────────────────────────────────
// PALETTE
//
// Change these constants to reskin the entire UI.
// ─────────────────────────────────────────────────────────────────────────────

/// Main background (toolbar, side panel, status bar).
pub const BG_METAL:   egui::Color32 = egui::Color32::from_rgb(0x8a, 0x8a, 0x88);
/// Darker background (title bar, pressed buttons).
pub const BG_DARK:    egui::Color32 = egui::Color32::from_rgb(0x7a, 0x7a, 0x78);
/// Dark panel / inset fill (settings sections, log area).
pub const PANEL_BG:   egui::Color32 = egui::Color32::from_rgb(0x3a, 0x3e, 0x3a);

/// Grid cell background.
pub const CELL_BG:    egui::Color32 = egui::Color32::from_rgb(0x2e, 0x32, 0x2e);
/// Grid cell fill (with tile loaded).
pub const CELL_FILL:  egui::Color32 = egui::Color32::from_rgb(0x34, 0x38, 0x34);
/// Grid cell selected highlight.
pub const CELL_SEL:   egui::Color32 = egui::Color32::from_rgb(0x4a, 0x5a, 0x4a);

/// Button face (default state).
pub const BTN_FACE:   egui::Color32 = egui::Color32::from_rgb(0x94, 0x94, 0x94);
/// Button highlight / bevel light edge.
pub const BTN_LIGHT:  egui::Color32 = egui::Color32::from_rgb(0xb8, 0xb8, 0xb8);
/// Button shadow / bevel dark edge.
pub const BTN_SHADOW: egui::Color32 = egui::Color32::from_rgb(0x5a, 0x5a, 0x5a);
/// Button text / dark foreground.
pub const BTN_TEXT:   egui::Color32 = egui::Color32::from_rgb(0x1a, 0x1a, 0x1a);

/// Dim text (title bar, status bar, section headers).
pub const TEXT_DIM:   egui::Color32 = egui::Color32::from_rgb(0x48, 0x48, 0x48);
/// LCD / light text on dark panels.
pub const TEXT_LCD:   egui::Color32 = egui::Color32::from_rgb(0xb0, 0xb0, 0xa8);

/// Inset border color.
pub const BORDER_IN:  egui::Color32 = egui::Color32::from_rgb(0x60, 0x60, 0x60);
/// Title bar grip dot highlight.
pub const GRIP_HI:    egui::Color32 = egui::Color32::from_rgb(0xb8, 0xb8, 0xb8);
/// Title bar grip dot shadow.
pub const GRIP_LO:    egui::Color32 = egui::Color32::from_rgb(0x5a, 0x5a, 0x5a);

// ─────────────────────────────────────────────────────────────────────────────
// BEVEL PRIMITIVES
//
// Low-level helpers for drawing raised and sunken 3D borders.
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a raised bevel (light top-left, dark bottom-right).
pub fn bevel_raised(ui: &egui::Ui, rect: egui::Rect) {
    ui.painter().line_segment([rect.left_top(), rect.right_top()],   egui::Stroke::new(1.0, BTN_LIGHT));
    ui.painter().line_segment([rect.left_top(), rect.left_bottom()], egui::Stroke::new(1.0, BTN_LIGHT));
    ui.painter().line_segment([rect.right_top(), rect.right_bottom()],  egui::Stroke::new(1.0, BTN_SHADOW));
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], egui::Stroke::new(1.0, BTN_SHADOW));
}

/// Draw a sunken bevel (dark top-left, light bottom-right).
pub fn bevel_sunken(ui: &egui::Ui, rect: egui::Rect) {
    ui.painter().line_segment([rect.left_top(), rect.right_top()],   egui::Stroke::new(1.0, BTN_SHADOW));
    ui.painter().line_segment([rect.left_top(), rect.left_bottom()], egui::Stroke::new(1.0, BTN_SHADOW));
    ui.painter().line_segment([rect.right_top(), rect.right_bottom()],  egui::Stroke::new(1.0, BTN_LIGHT));
    ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], egui::Stroke::new(1.0, BTN_LIGHT));
}

// ─────────────────────────────────────────────────────────────────────────────
// WIDGET HELPERS
//
// Pre-styled widgets that match the theme. All drawing is done here so that
// main.rs stays layout-only.
// ─────────────────────────────────────────────────────────────────────────────

/// Raised beveled button (toolbar style).
pub fn metal_button(ui: &mut egui::Ui, text: &str, enabled: bool) -> bool {
    let desired = egui::vec2(
        ui.painter().layout_no_wrap(text.to_string(), egui::FontId::monospace(11.0), BTN_TEXT).size().x + 20.0,
        20.0,
    );
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
    let hovered = response.hovered() && enabled;
    let pressed = response.is_pointer_button_down_on() && enabled;

    let bg = if pressed { BG_DARK } else if hovered { BTN_LIGHT } else { BTN_FACE };
    let fg = if enabled { BTN_TEXT } else { BORDER_IN };

    ui.painter().rect_filled(rect, 1.0, bg);
    if pressed { bevel_sunken(ui, rect); } else { bevel_raised(ui, rect); }

    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER,
        text, egui::FontId::monospace(11.0), fg);

    response.clicked() && enabled
}

/// Toggle / radio button (sunken when selected, raised otherwise).
/// If `fixed_width` is provided, the button uses that width instead of auto-sizing.
pub fn metal_toggle_sized(ui: &mut egui::Ui, text: &str, selected: bool, fixed_width: Option<f32>) -> bool {
    let w = fixed_width.unwrap_or_else(||
        ui.painter().layout_no_wrap(text.to_string(), egui::FontId::monospace(10.0), BTN_TEXT).size().x + 14.0
    );
    let desired = egui::vec2(w, 18.0);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());

    if selected {
        ui.painter().rect_filled(rect, 0.0, BG_DARK);
        bevel_sunken(ui, rect);
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::monospace(10.0), TEXT_LCD);
    } else {
        ui.painter().rect_filled(rect, 0.0, BTN_FACE);
        bevel_raised(ui, rect);
        ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::monospace(10.0), BTN_TEXT);
    }

    response.clicked()
}

/// Toggle / radio button with auto-sized width.
pub fn metal_toggle(ui: &mut egui::Ui, text: &str, selected: bool) -> bool {
    metal_toggle_sized(ui, text, selected, None)
}

/// Draw a row of toggle buttons that evenly divide the available width.
/// Returns the index of the clicked button (if any).
pub fn metal_toggle_row(ui: &mut egui::Ui, labels: &[&str], selected_idx: usize) -> Option<usize> {
    let avail = ui.available_width();
    let spacing = 2.0;
    let total_spacing = spacing * (labels.len() as f32 - 1.0);
    let btn_w = ((avail - total_spacing) / labels.len() as f32).floor();
    let mut clicked = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = spacing;
        for (i, label) in labels.iter().enumerate() {
            if metal_toggle_sized(ui, label, i == selected_idx, Some(btn_w)) {
                clicked = Some(i);
            }
        }
    });
    clicked
}

/// Section header with embossed text (e.g. "BLEND MARGIN").
pub fn section_header(ui: &mut egui::Ui, text: &str) {
    ui.add_space(6.0);
    let font = egui::FontId::monospace(11.0);
    let galley = ui.painter().layout_no_wrap(text.to_string(), font.clone(), BTN_TEXT);
    let desired = egui::vec2(ui.available_width(), galley.size().y + 2.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let pos = egui::pos2(rect.min.x + 2.0, rect.min.y);
    // Light shadow offset gives the embossed look
    ui.painter().text(pos + egui::vec2(1.0, 1.0), egui::Align2::LEFT_TOP, text, font.clone(), BTN_LIGHT);
    ui.painter().text(pos, egui::Align2::LEFT_TOP, text, font, BTN_TEXT);
    ui.add_space(2.0);
}

/// Dark inset panel with sunken 3D bevel (used for settings groups).
/// Stretches to fill available width with 6px horizontal padding.
pub fn inset_frame(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    let full_width = ui.available_width();
    let inner_w = full_width - 12.0; // 6px padding each side
    let frame = egui::Frame::new()
        .fill(PANEL_BG)
        .inner_margin(egui::Margin::symmetric(6, 4));
    let resp = frame.show(ui, |ui| {
        ui.set_min_width(inner_w);
        ui.set_max_width(inner_w);
        add_contents(ui);
    });
    // Clamp the bevel to full_width so it never overflows the panel
    let mut rect = resp.response.rect;
    rect.set_width(rect.width().min(full_width));
    bevel_sunken(ui, rect);
}

/// Checkbox in the Trilithium style with sunken bevel.
pub fn metal_checkbox(ui: &mut egui::Ui, checked: &mut bool, text: &str) {
    let desired = egui::vec2(ui.available_width(), 16.0);
    let (rect, response) = ui.allocate_exact_size(desired, egui::Sense::click());
    if response.clicked() { *checked = !*checked; }

    let box_rect = egui::Rect::from_min_size(
        egui::pos2(rect.min.x, rect.center().y - 6.0), egui::vec2(12.0, 12.0));
    ui.painter().rect_filled(box_rect, 0.0, BTN_FACE);
    bevel_sunken(ui, box_rect);

    if *checked {
        let pts = [
            egui::pos2(box_rect.min.x + 2.0, box_rect.center().y),
            egui::pos2(box_rect.center().x - 1.0, box_rect.max.y - 2.0),
            egui::pos2(box_rect.max.x - 1.0, box_rect.min.y + 2.0),
        ];
        ui.painter().line_segment([pts[0], pts[1]], egui::Stroke::new(2.0, BTN_TEXT));
        ui.painter().line_segment([pts[1], pts[2]], egui::Stroke::new(2.0, BTN_TEXT));
    }

    ui.painter().text(
        egui::pos2(box_rect.max.x + 6.0, rect.center().y),
        egui::Align2::LEFT_CENTER, text, egui::FontId::monospace(10.0), BTN_TEXT);
}

/// Raised header bar (for "SETTINGS", "PAINT TOOLS" titles).
pub fn panel_header(ui: &mut egui::Ui, text: &str) {
    let hdr_frame = egui::Frame::new().fill(BG_METAL)
        .inner_margin(egui::Margin::symmetric(0, 3));
    let hdr_resp = hdr_frame.show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new(text).monospace().size(11.0).strong().color(TEXT_DIM));
        });
    });
    bevel_raised(ui, hdr_resp.response.rect);
}

// ─────────────────────────────────────────────────────────────────────────────
// EGUI STYLE
//
// Call this once at startup to apply the theme to egui's built-in widgets
// (DragValue, TextEdit, ScrollArea, etc.).
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the Trilithium palette to egui's global style.
pub fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.override_font_id = Some(egui::FontId::monospace(11.0));
    style.spacing.item_spacing = egui::vec2(4.0, 3.0);
    style.spacing.button_padding = egui::vec2(6.0, 2.0);

    style.visuals.window_fill = BG_METAL;
    style.visuals.panel_fill = BG_METAL;
    style.visuals.extreme_bg_color = PANEL_BG;
    style.visuals.widgets.inactive.bg_fill = BTN_FACE;
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, BTN_TEXT);
    style.visuals.widgets.hovered.bg_fill = BTN_LIGHT;
    style.visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, BTN_TEXT);
    style.visuals.widgets.active.bg_fill = BG_DARK;
    style.visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, BTN_TEXT);
    style.visuals.widgets.noninteractive.bg_fill = PANEL_BG;
    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_LCD);
    style.visuals.selection.bg_fill = CELL_SEL;

    ctx.set_style(style);
}
